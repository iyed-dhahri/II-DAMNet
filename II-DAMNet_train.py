"""
II-DAMNet training — CLI version for cluster jobs.
  * preprocessing  : --version ours      -> letterbox + /255 (no normalization)   [default]
                     --version rescuenet  -> resize to CROP + ImageNet normalize
  * GT mapping     : YOUR confirmed mapping (raw3->intact, 4/5/6->damaged, 8/9->road, 255 ignore)
  * data           : ONLINE mix from 3 roots (trad/kind are images-only; labels shared in REAL/DATA)
        train : 100% real + 50% trad + 50% kind
        val   : --val-per-source from each (default 150)
  * monitoring     : every --vis-every epochs, overlays for fixed train & val samples
  * plots          : per-class IoU + mIoU (train & val) evolution, saved every epoch
  * resume         : --resume auto reloads <out>/last_<version>.pth on requeue
"""
import os, csv, random, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.pspnet import PSPNet

# ---- defaults (overridden by CLI in main) ----
REAL_ROOT = "rescue/DATA"      # real images + SHARED labels
TRAD_ROOT = "data_trad"        # images only
KIND_ROOT = "data_kind"        # images only
TRAIN_FRACTIONS = {"trad": 0.5, "kind": 0.5}
VAL_PER_SOURCE = 150
MIX_SEED = 0

CROP = 713
PREP = "letterbox"             # ours
NORMALIZE = False              # ours
NUM_CLASSES = 4
IGNORE_INDEX = 255
CLASS_NAMES = ["Intact", "Damaged", "Road", "Background"]
CLASS_RGB = np.array([[0, 255, 0], [255, 0, 0], [255, 255, 0], [0, 0, 0]], np.uint8)
IMN_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMN_STD  = np.array([0.229, 0.224, 0.225], np.float32)

OUT = "checkpoints/iidamnet2"
OVERLAY_EVERY = 2
N_OVERLAY = 4
VERSION = "ours"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== GT MAPPING (your confirmed mapping) =====================
def map_mask_to_4(m):
    out = np.full_like(m, 3, dtype=np.uint8)
    out[m == 3] = 0
    out[(m == 4) | (m == 5) | (m == 6)] = 1
    out[m == 8] = 2
    out[m == 255] = IGNORE_INDEX
    return out


# ===================== PREPROCESS =====================
def letterbox(img, lbl, size):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    lbl = cv2.resize(lbl, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ph, pw = size - nh, size - nw
    t, b, l, r = ph // 2, ph - ph // 2, pw // 2, pw - pw // 2
    img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
    lbl = cv2.copyMakeBorder(lbl, t, b, l, r, cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
    return img, lbl

def resize_prep(img, lbl, size):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    lbl = cv2.resize(lbl, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, lbl

def to_tensor(img):
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    if NORMALIZE:
        t = (t - torch.tensor(IMN_MEAN).view(3, 1, 1)) / torch.tensor(IMN_STD).view(3, 1, 1)
    return t


# ===================== DATA =====================
def resolve_label(stem, lbl_dir):
    base = stem
    for prefix in ("kind_", "trad_night_", "trad_"):
        if base.startswith(prefix):
            base = base[len(prefix):]; break
    for cand in (f"{base}_lab.png", f"{base.split('_')[-1]}_lab.png", f"{base}.png"):
        lp = os.path.join(lbl_dir, cand)
        if os.path.exists(lp):
            return lp
    return None

def collect_pairs(parent, split):
    img_dir = os.path.join(parent, split, "images")
    lbl_dir = os.path.join(REAL_ROOT, split, "labels")     # SHARED labels
    if not os.path.isdir(img_dir):
        print(f"  [warn] missing images dir {img_dir}"); return []
    if not os.path.isdir(lbl_dir):
        print(f"  [warn] missing shared label dir {lbl_dir}"); return []
    pairs, miss = [], 0
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lp = resolve_label(os.path.splitext(f)[0], lbl_dir)
        if lp: pairs.append((os.path.join(img_dir, f), lp))
        else:  miss += 1
    if miss:
        print(f"  [warn] {miss} images in {img_dir} had no matching label in {lbl_dir}")
    return pairs

def build_train_pairs(seed):
    rng = random.Random(seed)
    real = collect_pairs(REAL_ROOT, "train")
    trad = collect_pairs(TRAD_ROOT, "train")
    kind = collect_pairs(KIND_ROOT, "train")
    trad = rng.sample(trad, int(TRAIN_FRACTIONS["trad"] * len(trad)))
    kind = rng.sample(kind, int(TRAIN_FRACTIONS["kind"] * len(kind)))
    pairs = real + trad + kind
    rng.shuffle(pairs)
    print(f"  TRAIN mix: real={len(real)} trad={len(trad)} kind={len(kind)} -> {len(pairs)}")
    return pairs

def build_val_pairs(seed, n):
    rng = random.Random(seed + 1)
    out = []
    for tag, root in (("real", REAL_ROOT), ("trad", TRAD_ROOT), ("kind", KIND_ROOT)):
        p = collect_pairs(root, "val")
        take = rng.sample(p, min(n, len(p)))
        out += take
        print(f"  VAL {tag}: took {len(take)}/{len(p)}")
    return out

class SegDataset(Dataset):
    def __init__(self, pairs, crop): self.pairs, self.crop = pairs, crop
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ip, lp = self.pairs[i]
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        lbl = map_mask_to_4(np.array(Image.open(lp).convert("L")))
        img, lbl = (letterbox if PREP == "letterbox" else resize_prep)(img, lbl, self.crop)
        return to_tensor(img), torch.from_numpy(lbl.astype(np.int64))


# ===================== METRICS =====================
def add_hist(hist, pred, target):
    k = (target >= 0) & (target < NUM_CLASSES)
    hist += np.bincount(NUM_CLASSES * target[k].astype(int) + pred[k],
                        minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    return hist

def iou_per_class(hist):
    inter = np.diag(hist).astype(np.float64)
    union = hist.sum(1) + hist.sum(0) - inter
    return np.where(union > 0, inter / union, np.nan) * 100


# ===================== VISUALS =====================
def make_overlay(img01, mask, alpha=0.45):
    base = (img01 * 255).astype(np.float32)
    col = CLASS_RGB[np.clip(mask, 0, 3)].astype(np.float32)
    fg = (mask != 3) & (mask != IGNORE_INDEX)
    out = base.copy()
    for c in range(3):
        out[:, :, c] = np.where(fg, base[:, :, c] * (1 - alpha) + col[:, :, c] * alpha, base[:, :, c])
    return out.astype(np.uint8)

@torch.no_grad()
def save_overlays(model, dataset, indices, epoch, tag):
    model.eval()
    out_dir = os.path.join(OUT, "overlays", tag)
    os.makedirs(out_dir, exist_ok=True)
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.2 * n))
    if n == 1: axes = axes[None, :]
    for row, idx in enumerate(indices):
        x, y = dataset[idx]
        img01 = x.permute(1, 2, 0).numpy()
        if NORMALIZE:
            img01 = np.clip(img01 * IMN_STD + IMN_MEAN, 0, 1)
        gt = y.numpy()
        out = model(x.unsqueeze(0).to(device))
        out = out[0] if isinstance(out, (tuple, list)) else out
        out = F.interpolate(out, size=gt.shape[-2:], mode="bilinear", align_corners=True)
        pred = out.argmax(1).squeeze(0).cpu().numpy()
        axes[row, 0].imshow(img01);                     axes[row, 0].set_title("image")
        axes[row, 1].imshow(make_overlay(img01, gt));   axes[row, 1].set_title("ground truth")
        axes[row, 2].imshow(make_overlay(img01, pred)); axes[row, 2].set_title(f"pred ep{epoch+1}")
        for a in axes[row]: a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch+1:03d}.png"), dpi=130)
    plt.close()

def plot_curves(H, path):
    ep = range(1, len(H["train_loss"]) + 1)
    tr, va = np.array(H["train_iou"]), np.array(H["val_iou"])
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    ax[0, 0].plot(ep, H["train_loss"], "b-", label="train"); ax[0, 0].plot(ep, H["val_loss"], "r--", label="val")
    ax[0, 0].set_title("Loss"); ax[0, 0].set_xlabel("epoch"); ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)
    ax[0, 1].plot(ep, H["train_miou"], "b-", label="train mIoU"); ax[0, 1].plot(ep, H["val_miou"], "r--", label="val mIoU")
    ax[0, 1].plot(ep, H["train_miou_fg"], "b:", label="train excl bg"); ax[0, 1].plot(ep, H["val_miou_fg"], "r:", label="val excl bg")
    ax[0, 1].set_title("mIoU"); ax[0, 1].set_xlabel("epoch"); ax[0, 1].set_ylabel("%"); ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)
    for c in range(NUM_CLASSES):
        ax[1, 0].plot(ep, tr[:, c], label=CLASS_NAMES[c]); ax[1, 1].plot(ep, va[:, c], label=CLASS_NAMES[c])
    ax[1, 0].set_title("Train IoU per class"); ax[1, 0].set_xlabel("epoch"); ax[1, 0].set_ylabel("%"); ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)
    ax[1, 1].set_title("Val IoU per class");   ax[1, 1].set_xlabel("epoch"); ax[1, 1].set_ylabel("%"); ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def dump_csv(H, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["epoch", "train_loss", "val_loss", "train_miou", "val_miou", "train_miou_fg", "val_miou_fg"]
        head += [f"train_IoU_{c}" for c in CLASS_NAMES] + [f"val_IoU_{c}" for c in CLASS_NAMES]
        w.writerow(head)
        for i in range(len(H["train_loss"])):
            w.writerow([i + 1, H["train_loss"][i], H["val_loss"][i], H["train_miou"][i], H["val_miou"][i],
                        H["train_miou_fg"][i], H["val_miou_fg"][i]] + list(H["train_iou"][i]) + list(H["val_iou"][i]))


# ===================== MODEL =====================
def build_model(criterion):
    m = PSPNet(layers=101, bins=(1, 2, 3, 6), dropout=0.1, classes=4,
               zoom_factor=8, pretrained=False, criterion=criterion)
    m.layer0 = nn.Sequential(
        nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1))
    m.layer1[0].conv1 = nn.Conv2d(64, 64, 1, bias=False)
    m.layer1[0].downsample[0] = nn.Conv2d(64, 256, 1, bias=False)
    m.cls = nn.Sequential(nn.Conv2d(4096, 4, 1, bias=True))
    return m


# ===================== MAIN =====================
def main():
    global REAL_ROOT, TRAD_ROOT, KIND_ROOT, CROP, PREP, NORMALIZE
    global OUT, OVERLAY_EVERY, VAL_PER_SOURCE, VERSION

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="staged parent containing DATA/, data_trad/, data_kind/ (sets all 3 roots)")
    ap.add_argument("--real-root", default=REAL_ROOT)
    ap.add_argument("--trad-root", default=TRAD_ROOT)
    ap.add_argument("--kind-root", default=KIND_ROOT)
    ap.add_argument("--version", choices=["ours", "rescuenet"], default="ours")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--crop", type=int, default=713)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--val-per-source", type=int, default=150)
    ap.add_argument("--vis-every", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--resume", default="", help="'auto' to resume <out>/last_<version>.pth")
    a = ap.parse_args()

    REAL_ROOT, TRAD_ROOT, KIND_ROOT = a.real_root, a.trad_root, a.kind_root
    if a.data:
        REAL_ROOT = os.path.join(a.data, "DATA")
        TRAD_ROOT = os.path.join(a.data, "data_trad")
        KIND_ROOT = os.path.join(a.data, "data_kind")
    CROP, OUT, OVERLAY_EVERY, VAL_PER_SOURCE, VERSION = a.crop, a.out, a.vis_every, a.val_per_source, a.version
    PREP = "letterbox" if a.version == "ours" else "resize"
    NORMALIZE = (a.version == "rescuenet")
    os.makedirs(OUT, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    print(f"--- II-DAMNet | version={VERSION} (prep={PREP}, normalize={NORMALIZE}) | "
          f"crop={CROP} batch={a.batch} epochs={a.epochs} ---")
    print(f"roots: real={REAL_ROOT} trad={TRAD_ROOT} kind={KIND_ROOT}")

    train_ds = SegDataset(build_train_pairs(MIX_SEED), CROP)
    val_ds   = SegDataset(build_val_pairs(MIX_SEED, VAL_PER_SOURCE), CROP)
    train_ld = DataLoader(train_ds, batch_size=a.batch, shuffle=True, drop_last=True,
                          num_workers=a.workers, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=a.batch, shuffle=False,
                          num_workers=a.workers, pin_memory=True)
    tr_idx = list(np.linspace(0, len(train_ds) - 1, min(N_OVERLAY, len(train_ds))).astype(int))
    va_idx = list(np.linspace(0, len(val_ds) - 1, min(N_OVERLAY, len(val_ds))).astype(int))

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX).to(device)
    model = nn.DataParallel(build_model(criterion)).to(device)
    print(f"GPUs visible: {torch.cuda.device_count()}")

    optimizer = optim.AdamW(model.parameters(), lr=a.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    H = {k: [] for k in ["train_loss", "val_loss", "train_miou", "val_miou",
                         "train_miou_fg", "val_miou_fg", "train_iou", "val_iou"]}
    start_epoch, best = 0, 0.0
    last_path = os.path.join(OUT, f"last_{VERSION}.pth")
    best_path = os.path.join(OUT, f"best_{VERSION}.pth")
    if a.resume == "auto" and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optim"]); scheduler.load_state_dict(ck["sched"])
        start_epoch, best, H = ck["epoch"], ck.get("best", 0.0), ck.get("history", H)
        print(f"** resumed from {last_path} @ epoch {start_epoch} (best={best:.2f})")

    for epoch in range(start_epoch, a.epochs):
        model.train()
        tl, th = 0.0, np.zeros((NUM_CLASSES, NUM_CLASSES))
        for x, y in tqdm(train_ld, desc=f"ep{epoch+1}/{a.epochs} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, y)
            loss = out[1].mean() + 0.4 * out[2].mean()
            loss.backward(); optimizer.step()
            tl += loss.item()
            th = add_hist(th, out[0].cpu().numpy(), y.cpu().numpy())

        model.eval()
        vl, vh = 0.0, np.zeros((NUM_CLASSES, NUM_CLASSES))
        with torch.no_grad():
            for x, y in tqdm(val_ld, desc=f"ep{epoch+1}/{a.epochs} [val]"):
                x, y = x.to(device), y.to(device)
                out = model(x)
                out = out[0] if isinstance(out, (tuple, list)) else out
                out = F.interpolate(out, size=y.shape[-2:], mode="bilinear", align_corners=True)
                vl += criterion(out, y).item()
                vh = add_hist(vh, out.argmax(1).cpu().numpy(), y.cpu().numpy())
        scheduler.step()

        tr_iou, va_iou = iou_per_class(th), iou_per_class(vh)
        H["train_loss"].append(tl / max(1, len(train_ld))); H["val_loss"].append(vl / max(1, len(val_ld)))
        H["train_iou"].append(tr_iou); H["val_iou"].append(va_iou)
        H["train_miou"].append(np.nanmean(tr_iou)); H["val_miou"].append(np.nanmean(va_iou))
        H["train_miou_fg"].append(np.nanmean(tr_iou[:3])); H["val_miou_fg"].append(np.nanmean(va_iou[:3]))

        print(f"\n[ep {epoch+1}] train loss {H['train_loss'][-1]:.3f} mIoU {H['train_miou'][-1]:.2f} "
              f"| val loss {H['val_loss'][-1]:.3f} mIoU {H['val_miou'][-1]:.2f} (excl bg {H['val_miou_fg'][-1]:.2f})")
        print("  val IoU: " + " ".join(f"{CLASS_NAMES[c]}={va_iou[c]:.1f}" for c in range(NUM_CLASSES)))

        if (epoch + 1) % OVERLAY_EVERY == 0:
            save_overlays(model, train_ds, tr_idx, epoch, "train")
            save_overlays(model, val_ds, va_idx, epoch, "val")
            print(f"  overlays -> {OUT}/overlays/[train|val]/epoch_{epoch+1:03d}.png")

        plot_curves(H, os.path.join(OUT, f"curves_{VERSION}.png"))
        dump_csv(H, os.path.join(OUT, f"history_{VERSION}.csv"))

        torch.save({"state_dict": model.state_dict(), "optim": optimizer.state_dict(),
                    "sched": scheduler.state_dict(), "epoch": epoch + 1, "best": best,
                    "history": H}, last_path)
        if H["val_miou"][-1] > best:
            best = H["val_miou"][-1]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1, "best": best}, best_path)
            print(f"  ** new best val mIoU {best:.2f} -> {best_path}")

    print(f"\nDone. best val mIoU = {best:.2f} | {best_path}")


if __name__ == "__main__":
    main()