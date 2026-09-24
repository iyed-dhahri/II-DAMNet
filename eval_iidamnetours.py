"""
Evaluate ONLY the retrained II-DAMNet (ours) on the test sets, with sample overlays.

  model      : ResNet-101 PSPNet (models.pspnet) + head surgery, 4-class.
               trained with the 'ours' pipeline -> letterbox + /255 (NO normalization).
               native output order == canonical [intact, damaged, road, bg] -> identity perm.
  GT mapping : YOUR confirmed mapping (raw3->intact, 4/5/6->damaged, 8/9->road, else bg).
               -> must match training exactly, hence ROAD_IDS = [8, 9].
  overlays   : drawn on the ORIGINAL image (letterbox removed, pred resized back to native).
  scoring    : at 713 (letterbox, padding ignored) -> matches training-time validation.

Test sets (all share the original RescueNet test labels):
  rescuenet (<id>.jpg) | kind (kind_<id>) | synth (trad_night_<id>) | mix (online 150 each)
"""
import os, csv, re, random
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")            # use only physical GPU 1
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from collections import OrderedDict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from models.pspnet import PSPNet as ExtPSPNet

# ============================== CONFIG ==============================
MODEL_PATH = "checkpoints/iidamnet2/best_ours.pth"   # produced by II-DAMNet_train.py (--version ours)
LABEL_DIR  = "rescue/DATA/test/labels"            # shared RescueNet test labels

CROP        = 713
LIMIT       = None                                # None = all images, or int for a quick run
CSV_OUT     = "eval_iidamnet_ours.csv"

# --- preprocessing (ours) ---
PREPROCESS  = "letterbox"      # 'letterbox' (ours) or 'resize'
NORMALIZE   = False            # ours = /255 only
USE_BGR     = False
PRED_TO_CANON = [0, 1, 2, 3]   # trained directly in canonical order

# --- GT mapping (MUST match training) ---
ROAD_IDS = [8, 9]              # training used road = {8, 9}

# --- online mix ---
MIX_PER_SOURCE = 150
MIX_SEED       = 0

# --- overlays (samples) ---
SAVE_OVERLAY  = True
OVERLAY_DIR   = "overlays/II-DAMNet-ours"
MAX_OVERLAY   = 20             # samples saved per test set; None = all
ALPHA         = 0.5
DISPLAY_SIDE  = 1024           # longest side of saved overlay panels (keeps files small)
WITH_GT_PANEL = True           # panel = original | prediction | ground truth

NUM_CLASSES = 4
CLASS_NAMES = ["Building Intact", "Building Damaged", "Road Blocked", "Background"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TESTSETS = {
    "rescuenet": "rescue/DATA/test/images",
    "kind":      "data_kind/test/images",
    "synth":     "data_trad/test/images",
    # "mix" built online from the three above
}


# ===================== GT MAPPING (your confirmed mapping) =====================
def gt_map(mask):
    out = np.full_like(mask, 3, dtype=np.uint8)     # default -> background
    out[mask == 3] = 0                               # intact
    out[np.isin(mask, [4, 5, 6])] = 1                # damaged
    out[np.isin(mask, ROAD_IDS)] = 2                 # road blocked
    return out


# ===================== OVERLAY HELPERS =====================
COLORS = np.array([
    [0, 255, 0],     # 0 intact   -> green
    [255, 0, 0],     # 1 damaged  -> red
    [255, 255, 0],   # 2 road     -> yellow
    [0, 0, 0],       # 3 backgr   -> black (untinted)
], dtype=np.uint8)

def make_overlay(base_rgb, mask4):
    out = base_rgb.astype(np.float32).copy()
    cmask = COLORS[np.clip(mask4, 0, 3)].astype(np.float32)
    fg = mask4 != 3
    for c in range(3):
        out[:, :, c] = np.where(fg, base_rgb[:, :, c] * (1 - ALPHA) + cmask[:, :, c] * ALPHA,
                                base_rgb[:, :, c])
    return out.astype(np.uint8)

def to_display(orig, pred, gt, max_side):
    h, w = orig.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        dw, dh = int(w * s), int(h * s)
        orig = cv2.resize(orig, (dw, dh), interpolation=cv2.INTER_AREA)
        pred = cv2.resize(pred.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST)
        gt = cv2.resize(gt.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST)
    return orig, pred, gt

def save_panel(out_path, orig_rgb, pred_native, gt_native):
    orig, pred, gt = to_display(orig_rgb, pred_native, gt_native, DISPLAY_SIDE)
    pred_ov = make_overlay(orig, pred)
    if WITH_GT_PANEL:
        gt_ov = make_overlay(orig, gt)
        sep = np.full((orig.shape[0], 6, 3), 255, dtype=np.uint8)
        img = np.concatenate([orig, sep, pred_ov, sep, gt_ov], axis=1)
    else:
        img = pred_ov
    Image.fromarray(img).save(out_path)


# ===================== LETTERBOX (+ inverse) =====================
def letterbox_img(img, size):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ph, pw = size - nh, size - nw
    t, b, l, rr = ph // 2, ph - ph // 2, pw // 2, pw - pw // 2
    out = cv2.copyMakeBorder(r, t, b, l, rr, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return out, (t, l, nh, nw)

def letterbox_lbl(lbl, size, pad):
    h, w = lbl.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    r = cv2.resize(lbl, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ph, pw = size - nh, size - nw
    t, b, l, rr = ph // 2, ph - ph // 2, pw // 2, pw - pw // 2
    return cv2.copyMakeBorder(r, t, b, l, rr, cv2.BORDER_CONSTANT, value=int(pad))

def unletterbox(pred, geo, H0, W0):
    t, l, nh, nw = geo
    core = pred[t:t + nh, l:l + nw]
    return cv2.resize(core.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST)


def to_tensor(base_rgb):
    arr = base_rgb[:, :, ::-1] if USE_BGR else base_rgb
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0
    if NORMALIZE:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t - mean) / std
    return t


# ===================== MODEL (auto-detect arch from checkpoint) =====================
def _norm_state(ckpt):
    state = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    out = OrderedDict()
    for k, v in state.items():
        name = k[7:] if k.startswith('module.') else k
        if 'criterion' in name:
            continue
        out[name] = v
    return out

def build_and_load(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = _norm_state(ckpt)

    # backbone depth from number of blocks in layer3 (R50=6, R101=23, R152=36)
    idx = [int(m.group(1)) for k in state for m in [re.match(r"layer3\.(\d+)\.", k)] if m]
    n3 = (max(idx) + 1) if idx else 23
    layers = {6: 50, 23: 101, 36: 152}.get(n3, 101)

    # surgery stem has a single 7x7 conv (no second conv -> no 'layer0.3.weight')
    has_surgery = "layer0.3.weight" not in state
    # head: single 1x1 conv to NUM_CLASSES, or the full 3x3->512->1x1 head
    cls0 = state.get("cls.0.weight")
    single_cls = cls0 is not None and cls0.shape[0] == NUM_CLASSES and cls0.shape[-1] == 1
    print(f"  [detect] layers={layers} surgery_stem={has_surgery} single_conv_head={single_cls}")

    m = ExtPSPNet(layers=layers, bins=(1, 2, 3, 6), dropout=0.1,
                  classes=NUM_CLASSES, zoom_factor=8, pretrained=False)
    if has_surgery:
        m.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1))
        m.layer1[0].conv1 = nn.Conv2d(64, 64, 1, bias=False)
        m.layer1[0].downsample[0] = nn.Conv2d(64, 256, 1, bias=False)
    if single_cls:
        m.cls = nn.Sequential(nn.Conv2d(4096, NUM_CLASSES, 1, bias=True))
    # else keep the default full head (already classes=NUM_CLASSES)

    # load ONLY shape-matching tensors -> never raises on a mismatch
    shapes = {k: v.shape for k, v in m.state_dict().items()}
    filt, dropped = OrderedDict(), []
    for k, v in state.items():
        if k in shapes and tuple(v.shape) == tuple(shapes[k]):
            filt[k] = v
        else:
            dropped.append(k)
    miss, unexp = m.load_state_dict(filt, strict=False)
    print(f"  [load] loaded={len(filt)} dropped={len(dropped)} missing={len(miss)} unexpected={len(unexp)}")
    if dropped:
        print(f"         first dropped: {dropped[:4]}")
    if len(dropped) > 10:
        print("         !! many dropped tensors -> architecture likely still mismatched "
              "(wrong backbone family?). Check the [detect] line above.")
    return m.to(device).eval()


# ===================== GT FILE RESOLUTION & LISTING =====================
def find_gt(img_name):
    base = os.path.splitext(img_name)[0]
    for prefix in ("kind_", "trad_night_", "trad_"):
        if base.startswith(prefix):
            base = base[len(prefix):]; break
    for cand in (f"{base}_lab.png", f"{base.split('_')[-1]}_lab.png", f"{base}.png"):
        p = os.path.join(LABEL_DIR, cand)
        if os.path.exists(p):
            return p
    return None

def list_images(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')))

def build_mix():
    rng = random.Random(MIX_SEED)
    paths = []
    for name in ("rescuenet", "kind", "synth"):
        d = TESTSETS.get(name)
        if not d or not os.path.isdir(d):
            print(f"  [mix] source '{name}' dir missing ({d}) - skipped"); continue
        imgs = list_images(d)
        imgs = rng.sample(imgs, MIX_PER_SOURCE) if len(imgs) > MIX_PER_SOURCE else imgs
        paths += imgs
    return paths


# ===================== EVALUATE ONE TEST SET =====================
def evaluate(model, image_paths, t_name):
    perm = np.array(PRED_TO_CANON, dtype=np.uint8)
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    ov_dir = os.path.join(OVERLAY_DIR, t_name)
    if SAVE_OVERLAY:
        os.makedirs(ov_dir, exist_ok=True)

    n_eval, n_missing, n_ov = 0, 0, 0
    with torch.no_grad():
        for ip in tqdm(image_paths, leave=False, desc=t_name):
            img_name = os.path.basename(ip)
            gt_path = find_gt(img_name)
            if gt_path is None:
                n_missing += 1; continue
            bgr = cv2.imread(ip)
            if bgr is None:
                n_missing += 1; continue
            orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            H0, W0 = orig.shape[:2]
            gt_native = gt_map(np.array(Image.open(gt_path).convert('L')))
            # kind/synth images are 713x713 (stretched) while labels are native;
            # align the label to the image geometry so letterbox + overlay match.
            if gt_native.shape[:2] != (H0, W0):
                gt_native = cv2.resize(gt_native, (W0, H0), interpolation=cv2.INTER_NEAREST)

            # ---- model input + scoring labels at CROP ----
            if PREPROCESS == "letterbox":
                base, geo = letterbox_img(orig, CROP)
                gt_score = letterbox_lbl(gt_native, CROP, pad=255)   # pad ignored in scoring
            else:
                base = cv2.resize(orig, (CROP, CROP), interpolation=cv2.INTER_LINEAR); geo = None
                gt_score = cv2.resize(gt_native, (CROP, CROP), interpolation=cv2.INTER_NEAREST)

            x = to_tensor(base).unsqueeze(0).to(device)
            out = model(x)
            out = out[0] if isinstance(out, tuple) else out
            if out.shape[-2:] != (CROP, CROP):
                out = F.interpolate(out, (CROP, CROP), mode="bilinear", align_corners=True)
            pred713 = perm[out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)]

            k = (gt_score >= 0) & (gt_score < NUM_CLASSES)          # excludes 255 padding
            hist += np.bincount(NUM_CLASSES * gt_score[k].astype(int) + pred713[k],
                                minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
            n_eval += 1

            # ---- overlay on the ORIGINAL image ----
            if SAVE_OVERLAY and (MAX_OVERLAY is None or n_ov < MAX_OVERLAY):
                pred_native = unletterbox(pred713, geo, H0, W0) if geo else \
                              cv2.resize(pred713, (W0, H0), interpolation=cv2.INTER_NEAREST)
                save_panel(os.path.join(ov_dir, f"overlay_{os.path.splitext(img_name)[0]}.png"),
                           orig, pred_native, gt_native)
                n_ov += 1

    inter = np.diag(hist).astype(np.float64)
    union = hist.sum(1) + hist.sum(0) - inter
    iou = np.where(union > 0, inter / union, np.nan) * 100
    return iou, np.nanmean(iou), np.nanmean(iou[:3]), n_eval, n_missing


# ===================== MAIN =====================
def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"weights not found: {MODEL_PATH}")
    print(f"Loading II-DAMNet ours: {MODEL_PATH} | prep={PREPROCESS} normalize={NORMALIZE} "
          f"road_ids={ROAD_IDS}")
    model = build_and_load(MODEL_PATH)

    sets = {}
    for name, d in TESTSETS.items():
        if os.path.isdir(d):
            sets[name] = list_images(d)
        else:
            print(f"-- testset '{name}': dir '{d}' not found, skipping.")
    sets["mix"] = build_mix()
    if LIMIT:
        sets = {k: v[:LIMIT] for k, v in sets.items()}

    rows = []
    for t_name, paths in sets.items():
        if not paths:
            continue
        iou, miou_all, miou_fg, n_eval, n_missing = evaluate(model, paths, t_name)
        print(f"\n=== II-DAMNet ours on '{t_name}'  ({n_eval} imgs, {n_missing} GT-missing) ===")
        for i, name in enumerate(CLASS_NAMES):
            v = f"{iou[i]:6.2f}%" if not np.isnan(iou[i]) else "   n/a"
            print(f"   {name:<18}: {v}")
        print(f"   {'mIoU incl. bg':<18}: {miou_all:6.2f}%")
        print(f"   {'mIoU excl. bg':<18}: {miou_fg:6.2f}%")
        if SAVE_OVERLAY:
            print(f"   overlays -> {os.path.join(OVERLAY_DIR, t_name)}/")
        rows.append({
            "testset": t_name, "n_images": n_eval,
            "IoU_intact": f"{iou[0]:.2f}", "IoU_damaged": f"{iou[1]:.2f}",
            "IoU_road": f"{iou[2]:.2f}", "IoU_background": f"{iou[3]:.2f}",
            "mIoU_incl_bg": f"{miou_all:.2f}", "mIoU_excl_bg": f"{miou_fg:.2f}",
        })

    print("\n" + "=" * 70)
    print(f"{'testset':<12}{'mIoU incl bg':>16}{'mIoU excl bg':>16}{'imgs':>10}")
    print("-" * 70)
    for r in rows:
        print(f"{r['testset']:<12}{r['mIoU_incl_bg']:>15}%{r['mIoU_excl_bg']:>15}%{r['n_images']:>10}")
    print("=" * 70)

    if rows:
        with open(CSV_OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nSaved results to '{CSV_OUT}'")


if __name__ == "__main__":
    main()