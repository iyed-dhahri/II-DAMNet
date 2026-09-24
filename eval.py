"""
Unified test-set evaluation, scored on 4 canonical classes:
    0 = Building Intact | 1 = Building Damaged | 2 = Road Blocked | 3 = Background

Models compared
---------------
  II-DAMNet         : ResNet-101 PSPNet (models.pspnet) + head surgery, 4-class.
                      Native output order [intact, damaged, road, bg].
                      preprocess = resize, ImageNet-normalized, RGB.
  PSPNet-ours       : ResNet-50 PSPNet (defined here), 4-class, "ours" version.
                      Native output order [bg, intact, damaged, road] (MACRO_NAMES).
                      preprocess = letterbox, NO normalization (/255 only).
  PSPNet-rescuenet  : same arch, "rescuenet" version.
                      preprocess = resize, ImageNet-normalized.

Test sets (all share the ORIGINAL RescueNet labels)
  rescuenet : original images   (<id>.jpg)        -> GT <id>_lab.png
  kind      : KinD night images (kind_<id>.png)   -> GT <id>_lab.png
  synth     : synth night       (trad_night_<id>) -> GT <id>_lab.png
  mix       : built ONLINE = 150 rescuenet + 150 kind + 150 synth (no folder needed)

IMPORTANT - class order / GT mapping
------------------------------------
Each model emits predictions in ITS OWN class order; we permute every prediction
into the canonical order above via cfg["pred_to_canon"]. The GROUND TRUTH is
mapped from raw RescueNet ids with cfg["gt_map"] (default "standard"). Your two
original scripts used different GT mappings (see GT_MAPS below) - if a model was
trained on the other convention its overlays will look shifted; switch its
"gt_map" then.
"""

import os
import csv
import random
from collections import OrderedDict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm

from models.pspnet import PSPNet as ExtPSPNet   # the ResNet-101 net used by II-DAMNet

# ============================== CONFIG ==============================
LABEL_DIR = "rescue/DATA/test/labels"           # shared RescueNet labels

CROP        = 713
LIMIT       = None                              # None = all, or int for a smoke test
CSV_OUT     = "eval_matrix_results.csv"

# --- online mix ---
MIX_PER_SOURCE = 150
MIX_SEED       = 0

# --- overlays ---
SAVE_OVERLAY  = True
OVERLAY_DIR   = "overlays"
MAX_OVERLAY   = 30                              # per (model, testset); None = all
ALPHA         = 0.5
WITH_GT_PANEL = True                            # original | prediction | GT

# RescueNet ids counted as "Road Blocked" (standard GT mapping).
ROAD_IDS = [8]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

MODELS = {
    "II-DAMNet": {
        "arch":         "pspnet_r101",
        "path":         "checkpoints/II-DAMNet/II-DAMNet.pth",
        "preprocess":   "resize",
        "normalize":    True,
        "use_bgr":      False,
        "pred_to_canon": [0, 1, 2, 3],          # native [intact,damaged,road,bg] == canonical
        "gt_map":       "standard",
    },
    "PSPNet-ours": {
        "arch":         "pspnet_r50",
        "path":         "checkpoints/pspnet/best_ours.pth",      # <-- adjust
        "preprocess":   "letterbox",
        "normalize":    False,
        "use_bgr":      False,
        "pred_to_canon": [3, 0, 1, 2],          # native [bg,intact,damaged,road] -> canonical
        "gt_map":       "standard",
    },
    "PSPNet-rescuenet": {
        "arch":         "pspnet_r50",
        "path":         "checkpoints/pspnet/best_rescuenet.pth", # <-- adjust
        "preprocess":   "resize",
        "normalize":    True,
        "use_bgr":      False,
        "pred_to_canon": [3, 0, 1, 2],
        "gt_map":       "standard",
    },
}

TESTSETS = {
    "rescuenet": "rescue/DATA/test/images",
    "kind":      "data_kind/test/images",
    "synth":     "data_trad/test/images",
    # "mix" is built online from the three above -> no folder
}

NUM_CLASSES = 4
CLASS_NAMES = ["Building Intact", "Building Damaged", "Road Blocked", "Background"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================== GROUND-TRUTH MAPPINGS (raw RescueNet id -> canonical) =====================
# canonical: 0 intact, 1 damaged, 2 road, 3 background
def gt_standard(mask):
    out = np.full_like(mask, 3, dtype=np.uint8)
    out[mask == 3] = 0                          # building no-damage  -> intact
    out[np.isin(mask, [4, 5, 6])] = 1           # minor/major/total   -> damaged
    out[np.isin(mask, ROAD_IDS)] = 2            # road blocked
    return out

def gt_testeval(mask):
    # The mapping from test_eval_pspnet.py's to_macro, re-expressed in canonical order.
    # (their to_macro: raw3->intact, raw4/5/6->damaged, raw8->road)
    out = np.full_like(mask, 3, dtype=np.uint8)
    out[mask == 3] = 0
    out[np.isin(mask, [4, 5, 6])] = 1
    out[mask == 8] = 2
    return out

GT_MAPS = {"standard": gt_standard, "testeval": gt_testeval}


# ===================== OVERLAY HELPERS =====================
COLORS = np.array([
    [0, 255, 0],     # 0 intact   -> green
    [255, 0, 0],     # 1 damaged  -> red
    [255, 255, 0],   # 2 road     -> yellow
    [0, 0, 0],       # 3 backgr   -> black (untinted)
], dtype=np.uint8)

def make_overlay(base_rgb, mask4):
    out = base_rgb.astype(np.float32).copy()
    cmask = COLORS[mask4].astype(np.float32)
    fg = mask4 != 3
    for c in range(3):
        out[:, :, c] = np.where(
            fg, base_rgb[:, :, c] * (1 - ALPHA) + cmask[:, :, c] * ALPHA, base_rgb[:, :, c])
    return out.astype(np.uint8)

def save_panel(out_path, base_rgb, pred4, gt4=None):
    pred_ov = make_overlay(base_rgb, pred4)
    if gt4 is not None and WITH_GT_PANEL:
        gt_ov = make_overlay(base_rgb, gt4)
        sep = np.full((base_rgb.shape[0], 6, 3), 255, dtype=np.uint8)
        img = np.concatenate([base_rgb, sep, pred_ov, sep, gt_ov], axis=1)
    else:
        img = pred_ov
    Image.fromarray(img).save(out_path)


# ===================== PREPROCESSING (must match each model's training) =====================
def letterbox(img, mask, size, pad_label):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ph, pw = size - nh, size - nw
    t, b, l, r = ph // 2, ph - ph // 2, pw // 2, pw - pw // 2
    img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
    mask = cv2.copyMakeBorder(mask, t, b, l, r, cv2.BORDER_CONSTANT, value=int(pad_label))
    return img, mask

def preprocess(orig_rgb, gt_canon, mode, size):
    """Returns (base_rgb[size,size], gt_canon[size,size]) consistently transformed."""
    if mode == "letterbox":
        return letterbox(orig_rgb, gt_canon, size, pad_label=3)   # pad with canonical background
    img = cv2.resize(orig_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt_canon, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, gt

def to_tensor(base_rgb, normalize, use_bgr):
    arr = base_rgb[:, :, ::-1] if use_bgr else base_rgb
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0
    if normalize:
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t


# ===================== ARCHITECTURES =====================
# ----- ResNet-50 PSPNet (ported verbatim from test_eval_pspnet.py) -----
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins=(1, 2, 3, 6)):
        super().__init__()
        self.features = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(b),
                          nn.Conv2d(in_dim, reduction_dim, 1, bias=False),
                          nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True))
            for b in bins])
    def forward(self, x):
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), size=x.shape[2:], mode="bilinear", align_corners=True))
        return torch.cat(out, 1)

class PSPNetR50(nn.Module):
    def __init__(self, n_classes=4, pretrained=False, bins=(1, 2, 3, 6), dropout=0.1):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet50(weights=weights,
                                              replace_stride_with_dilation=[False, True, True])
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2 = resnet.layer1, resnet.layer2
        self.layer3, self.layer4 = resnet.layer3, resnet.layer4
        self.ppm = PPM(2048, 512, bins)
        self.cls = nn.Sequential(
            nn.Conv2d(2048 + 512 * len(bins), 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout), nn.Conv2d(512, n_classes, 1))
        self.aux = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout), nn.Conv2d(256, n_classes, 1))
    def forward(self, x):
        h, w = x.shape[2:]
        x = self.layer0(x); x = self.layer1(x); x = self.layer2(x)
        x3 = self.layer3(x); x4 = self.layer4(x3)
        out = self.cls(self.ppm(x4))
        out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
        if self.training:
            aux = F.interpolate(self.aux(x3), (h, w), mode="bilinear", align_corners=True)
            return out, aux
        return out

def build_r101_surgery(n_classes=4):
    m = ExtPSPNet(layers=101, bins=(1, 2, 3, 6), dropout=0.1,
                  classes=n_classes, zoom_factor=8, pretrained=False)
    m.layer0 = nn.Sequential(
        nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1))
    m.layer1[0].conv1 = nn.Conv2d(64, 64, 1, bias=False)
    m.layer1[0].downsample[0] = nn.Conv2d(64, 256, 1, bias=False)
    m.cls = nn.Sequential(nn.Conv2d(4096, n_classes, 1, bias=True))
    return m

def build_model(arch):
    if arch == "pspnet_r101":
        return build_r101_surgery(4)
    if arch == "pspnet_r50":
        return PSPNetR50(4, pretrained=False)
    raise ValueError(f"unknown arch '{arch}'")


def load_weights(model, path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    new_state = OrderedDict()
    for k, v in state.items():
        name = k[7:] if k.startswith('module.') else k
        if 'criterion' in name:
            continue
        new_state[name] = v
    miss, unexp = model.load_state_dict(new_state, strict=False)
    if miss or unexp:
        print(f"  [load] missing={len(miss)} unexpected={len(unexp)} "
              f"(first missing: {miss[:2]})")
    return model.to(device).eval()


# ===================== EVALUATE ONE (model, testset) =====================
def evaluate(model, cfg, image_paths, m_name, t_name):
    gt_fn = GT_MAPS[cfg["gt_map"]]
    pred_perm = np.array(cfg["pred_to_canon"], dtype=np.uint8)
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    ov_dir = None
    if SAVE_OVERLAY:
        ov_dir = os.path.join(OVERLAY_DIR, m_name, t_name)
        os.makedirs(ov_dir, exist_ok=True)

    n_eval, n_missing, n_ov = 0, 0, 0
    with torch.no_grad():
        for ip in tqdm(image_paths, leave=False, desc=f"{m_name}/{t_name}"):
            img_name = os.path.basename(ip)
            gt_path = find_gt(img_name)
            if gt_path is None:
                n_missing += 1
                continue

            bgr = cv2.imread(ip)
            if bgr is None:
                n_missing += 1
                continue
            orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            gt_raw = np.array(Image.open(gt_path).convert('L'))
            gt_canon_full = gt_fn(gt_raw)

            base, gt4 = preprocess(orig, gt_canon_full, cfg["preprocess"], CROP)
            x = to_tensor(base, cfg["normalize"], cfg["use_bgr"]).unsqueeze(0).to(device)

            out = model(x)
            out = out[0] if isinstance(out, tuple) else out
            if out.shape[-2:] != (CROP, CROP):
                out = F.interpolate(out, (CROP, CROP), mode="bilinear", align_corners=True)
            pred = out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred4 = pred_perm[pred]                       # -> canonical order

            k = (gt4 >= 0) & (gt4 < NUM_CLASSES)
            hist += np.bincount(NUM_CLASSES * gt4[k].astype(int) + pred4[k],
                                minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
            n_eval += 1

            if SAVE_OVERLAY and (MAX_OVERLAY is None or n_ov < MAX_OVERLAY):
                save_panel(os.path.join(ov_dir, f"overlay_{os.path.splitext(img_name)[0]}.png"),
                           base, pred4, gt4)
                n_ov += 1

    inter = np.diag(hist).astype(np.float64)
    union = hist.sum(1) + hist.sum(0) - inter
    iou = np.where(union > 0, inter / union, np.nan) * 100
    return iou, np.nanmean(iou), np.nanmean(iou[:3]), n_eval, n_missing


# ===================== GT FILE RESOLUTION & FILE LISTING =====================
def find_gt(img_name):
    base = os.path.splitext(img_name)[0]
    for prefix in ("kind_", "trad_night_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
    for cand in (f"{base}_lab.png", f"{base.split('_')[-1]}_lab.png", f"{base}.png"):
        p = os.path.join(LABEL_DIR, cand)
        if os.path.exists(p):
            return p
    return None

def list_images(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')))

def build_mix():
    """Online mix: MIX_PER_SOURCE images sampled from each of the 3 real test sets."""
    rng = random.Random(MIX_SEED)
    paths = []
    for name in ("rescuenet", "kind", "synth"):
        d = TESTSETS.get(name)
        if not d or not os.path.isdir(d):
            print(f"  [mix] source '{name}' dir missing ({d}) - skipped")
            continue
        imgs = list_images(d)
        if len(imgs) > MIX_PER_SOURCE:
            imgs = rng.sample(imgs, MIX_PER_SOURCE)
        else:
            print(f"  [mix] '{name}' has only {len(imgs)} (< {MIX_PER_SOURCE}); taking all")
        paths += imgs
    return paths


# ===================== MAIN =====================
def gather_testset_paths():
    sets = {}
    for name, d in TESTSETS.items():
        if os.path.isdir(d):
            sets[name] = list_images(d)
        else:
            print(f"-- testset '{name}': dir '{d}' not found, skipping.")
    sets["mix"] = build_mix()           # built online from the three above
    return sets

def main():
    testset_paths = gather_testset_paths()
    if LIMIT:
        testset_paths = {k: v[:LIMIT] for k, v in testset_paths.items()}

    rows = []
    for m_name, cfg in MODELS.items():
        if not os.path.exists(cfg["path"]):
            print(f"!! {m_name}: weights '{cfg['path']}' not found, skipping.")
            continue
        print(f"\n############ MODEL: {m_name}  (arch={cfg['arch']}, prep={cfg['preprocess']}, "
              f"norm={cfg['normalize']}, gt={cfg['gt_map']}) ############")
        model = load_weights(build_model(cfg["arch"]), cfg["path"])

        for t_name, paths in testset_paths.items():
            if not paths:
                continue
            iou, miou_all, miou_fg, n_eval, n_missing = evaluate(model, cfg, paths, m_name, t_name)

            print(f"\n  === {m_name} on '{t_name}'  ({n_eval} imgs, {n_missing} GT-missing) ===")
            for i, name in enumerate(CLASS_NAMES):
                val = f"{iou[i]:6.2f}%" if not np.isnan(iou[i]) else "   n/a"
                print(f"    {name:<18}: {val}")
            print(f"    {'mIoU incl. bg':<18}: {miou_all:6.2f}%")
            print(f"    {'mIoU excl. bg':<18}: {miou_fg:6.2f}%")

            rows.append({
                "model": m_name, "testset": t_name, "n_images": n_eval,
                "IoU_intact":   f"{iou[0]:.2f}", "IoU_damaged":   f"{iou[1]:.2f}",
                "IoU_road":     f"{iou[2]:.2f}", "IoU_background": f"{iou[3]:.2f}",
                "mIoU_incl_bg": f"{miou_all:.2f}", "mIoU_excl_bg": f"{miou_fg:.2f}",
            })

    print("\n" + "=" * 82)
    print(" SUMMARY  (mIoU)")
    print("=" * 82)
    print(f"{'model':<20}{'testset':<12}{'mIoU incl bg':>14}{'mIoU excl bg':>14}{'imgs':>8}")
    print("-" * 82)
    for r in rows:
        print(f"{r['model']:<20}{r['testset']:<12}"
              f"{r['mIoU_incl_bg']:>13}%{r['mIoU_excl_bg']:>13}%{r['n_images']:>8}")
    print("=" * 82)

    if rows:
        with open(CSV_OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nSaved detailed results to '{CSV_OUT}'")


if __name__ == "__main__":
    main()