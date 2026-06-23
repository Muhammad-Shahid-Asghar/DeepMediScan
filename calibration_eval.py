# ============================================================
# DeepMediScan - calibration_eval.py
# ------------------------------------------------------------
# Recomputes confidence-calibration metrics on the NEW leakage-safe
# checkpoints + leakage-safe Val/Test splits, so Section 8.2 of the
# report can report calibration on the same clean data as everything else.
#
# For EACH model (EfficientNet-B0/B4, ResNet-50, MobileNetV3-Large)
# and EACH split (Val, Test) it reports:
#   - ECE  : Expected Calibration Error (15 equal-width confidence bins)
#   - MCE  : Maximum Calibration Error (worst bin gap) -- extra context
#   - Brier: multi-class Brier score (mean squared error vs one-hot)
#   - Mean predicted confidence vs. actual accuracy (over/under-confidence)
#
# ECE definition (standard, Guo et al. 2017):
#   Bin the top-1 confidence into M bins. For each bin, take the gap
#   between average confidence and accuracy, weight by bin size, sum.
#       ECE = sum_m (|B_m|/N) * | acc(B_m) - conf(B_m) |
#   Lower is better. ECE < 0.10 is generally "reasonably calibrated".
#
# Brier (multi-class):
#   mean over samples of sum_k (p_k - y_k)^2,  y = one-hot.
#
# Model building + checkpoint loading is COPIED from main.py (same as the
# other eval scripts), so checkpoints load unchanged and the class order
# is the verified ground truth.
#
# Reproducible: fixed SEED (default 42).
#
# USAGE:
#   1. Confirm paths in the CONFIG block at the BOTTOM.
#   2. python calibration_eval.py
#   3. Paste the CALIBRATION TABLE into the chat so Section 8.2 can be updated.
#      Machine-readable copy: calibration_results.json
# ============================================================

import os
import json
import numpy as np

import torch
import torch.nn as nn
import timm

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ------------------------------------------------------------
# 0. Reproducibility
# ------------------------------------------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. MODEL ARCHITECTURE  (copied verbatim from main.py)
# ============================================================
class SkinModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.dropout  = nn.Dropout(0.4)
        self.backbone.eval()
        with torch.no_grad():
            dummy           = torch.zeros(1, 3, 224, 224)
            actual_features = self.backbone(dummy).shape[1]
        self.backbone.train()
        self.classifier = nn.Linear(actual_features, num_classes)

    def forward(self, x):
        return self.classifier(self.dropout(self.backbone(x)))


# ============================================================
# 2. PREPROCESSING  (copied verbatim from main.py)
# ============================================================
def get_transform(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def load_image_tensor(path, img_size, device):
    np_img = np.array(Image.open(path).convert("RGB"))
    return get_transform(img_size)(image=np_img)["image"].unsqueeze(0).to(device)


# ============================================================
# 3. CHECKPOINT LOADING  (mirrors main.py)
# ============================================================
def resolve_checkpoint_path(candidate_paths):
    for p in candidate_paths:
        if p and os.path.exists(p):
            return p
    return None


def load_model_and_classes(model_name, ckpt_path, device, fallback_classes):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_classes  = checkpoint.get("num_classes", len(fallback_classes))

    model = SkinModel(model_name=model_name, num_classes=n_classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    c2i = checkpoint.get("class_to_idx", None)
    if c2i and len(c2i) == n_classes:
        classes = [k for k, _ in sorted(c2i.items(), key=lambda kv: kv[1])]
    else:
        classes = list(fallback_classes[:n_classes])
    return model, classes, n_classes


# ============================================================
# 4. DATASET SCAN
# ============================================================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def scan_split(split_dir, class_order):
    name_to_idx = {c: i for i, c in enumerate(class_order)}
    paths, labels = [], []
    for cname in class_order:
        cdir = os.path.join(split_dir, cname)
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(cdir, fn))
                labels.append(name_to_idx[cname])
    return paths, np.array(labels, dtype=np.int64)


# ============================================================
# 5. INFERENCE  ->  probability matrix
# ============================================================
@torch.no_grad()
def predict_probs(model, paths, img_size, device, log_every=500):
    rows = []
    for i, p in enumerate(paths):
        logits = model(load_image_tensor(p, img_size, device))
        rows.append(torch.softmax(logits, dim=1)[0].cpu().numpy())
        if log_every and (i + 1) % log_every == 0:
            print(f"      ... {i + 1}/{len(paths)}", flush=True)
    return np.vstack(rows)


# ============================================================
# 6. CALIBRATION METRICS
# ============================================================
def expected_calibration_error(probs, y_true, n_bins=15):
    """
    ECE + MCE using top-1 confidence, M equal-width bins on [0,1].
    Returns (ece, mce, bin_table) where bin_table is a list of dicts.
    """
    conf = probs.max(axis=1)            # top-1 confidence
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    N = len(y_true)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0
    table = []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        # last bin includes the right edge
        if b == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum())
        if count == 0:
            table.append({"bin": f"[{lo:.2f},{hi:.2f})", "count": 0,
                          "avg_conf": None, "accuracy": None, "gap": None})
            continue
        avg_conf = float(conf[mask].mean())
        acc      = float(correct[mask].mean())
        gap      = abs(acc - avg_conf)
        ece += (count / N) * gap
        mce  = max(mce, gap)
        table.append({"bin": f"[{lo:.2f},{hi:.2f})", "count": count,
                      "avg_conf": round(avg_conf, 4),
                      "accuracy": round(acc, 4),
                      "gap": round(gap, 4)})
    return float(ece), float(mce), table


def brier_score_multiclass(probs, y_true, n_classes):
    """Mean over samples of sum_k (p_k - onehot_k)^2."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def evaluate_calibration(probs, y_true, n_classes, n_bins=15):
    ece, mce, table = expected_calibration_error(probs, y_true, n_bins)
    brier = brier_score_multiclass(probs, y_true, n_classes)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc  = float((pred == y_true).mean())
    mean_conf = float(conf.mean())
    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier": round(brier, 4),
        "accuracy": round(acc, 4),
        "mean_confidence": round(mean_conf, 4),
        "over_confidence": round(mean_conf - acc, 4),  # +ve = over-confident
        "n_images": int(len(y_true)),
        "bins": table,
    }


# ============================================================
# 7. MAIN
# ============================================================
def main(cfg):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nSeed: {cfg['seed']}\nECE bins: {cfg['n_bins']}")

    all_results = {}

    for model_name, cands in cfg["model_files"].items():
        ckpt = resolve_checkpoint_path(cands)
        if ckpt is None:
            print(f"\nSKIP {model_name}: no checkpoint found in {cands}")
            continue

        print("\n" + "=" * 64)
        print(f"MODEL: {model_name}   ckpt={os.path.basename(ckpt)}")
        model, classes, n_classes = load_model_and_classes(
            model_name, ckpt, device, cfg["fallback_classes"])
        img_size = cfg["img_sizes"].get(model_name, 224)
        all_results[model_name] = {}

        for split_name, split_dir in cfg["splits"].items():
            if not os.path.isdir(split_dir):
                print(f"  [{split_name}] folder not found: {split_dir} — skipping")
                continue
            paths, y_true = scan_split(split_dir, classes)
            print(f"  [{split_name}] {len(paths)} images — running inference...")
            probs = predict_probs(model, paths, img_size, device)
            res = evaluate_calibration(probs, y_true, n_classes, cfg["n_bins"])
            all_results[model_name][split_name] = res
            print(f"  [{split_name}] ECE={res['ece']:.4f}  MCE={res['mce']:.4f}  "
                  f"Brier={res['brier']:.4f}  "
                  f"meanConf={res['mean_confidence']:.4f} vs Acc={res['accuracy']:.4f} "
                  f"(over-conf {res['over_confidence']:+.4f})")

    # ---- Summary table ----
    print("\n" + "=" * 92)
    print("CALIBRATION TABLE (leakage-safe split) — for Section 8.2")
    print("=" * 92)
    print(f"{'Model':<24} {'Split':<6} {'ECE':>7} {'MCE':>7} {'Brier':>8} "
          f"{'MeanConf':>9} {'Accuracy':>9} {'Over-conf':>10}")
    print("-" * 92)
    for model_name, splits in all_results.items():
        for split_name, r in splits.items():
            print(f"{model_name:<24} {split_name:<6} "
                  f"{r['ece']:7.4f} {r['mce']:7.4f} {r['brier']:8.4f} "
                  f"{r['mean_confidence']:9.4f} {r['accuracy']:9.4f} "
                  f"{r['over_confidence']:+10.4f}")
    print("-" * 92)
    print("ECE/MCE/Brier: lower is better. ECE < 0.10 = reasonably calibrated.")
    print("Over-conf > 0 means the model's confidence exceeds its accuracy.")

    out = cfg.get("out_json", "calibration_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved machine-readable results to: {out}")
    print("\nDONE. Paste the CALIBRATION TABLE block into the chat.")


# ============================================================
# 8. CONFIG  ——  EDIT PATHS, THEN: python calibration_eval.py
# ============================================================
if __name__ == "__main__":
    DERMAI_DIR    = r"C:\Users\M Shahid Asghar\Disease Dectection Project\DermAI"
    RETRAINED_DIR = os.path.join(DERMAI_DIR, "retrained_models")

    CONFIG = {
        "seed":   42,
        "n_bins": 15,        # ECE bins (Guo et al. 2017 use 15)

        "model_files": {
            "efficientnet_b0": [
                os.path.join(RETRAINED_DIR, "best_efficientnet_b0_22class.pth"),
            ],
            "efficientnet_b4": [
                os.path.join(RETRAINED_DIR, "best_efficientnet_b4_22class.pth"),
            ],
            "resnet50": [
                os.path.join(RETRAINED_DIR, "best_resnet50_22class.pth"),
            ],
            "mobilenetv3_large_100": [
                os.path.join(RETRAINED_DIR, "best_mobilenetv3_large_100_22class.pth"),
                os.path.join(RETRAINED_DIR, "best_mobilenetv3_22class.pth"),
            ],
        },

        "img_sizes": {
            "efficientnet_b0":       224,
            "efficientnet_b4":       224,
            "resnet50":              224,
            "mobilenetv3_large_100": 224,
        },

        "splits": {
            "Val":  os.path.join(DERMAI_DIR, "SkinDisease_dataset_resplit", "resplit_output", "Val"),
            "Test": os.path.join(DERMAI_DIR, "SkinDisease_dataset_resplit", "resplit_output", "Test"),
        },

        "fallback_classes": [
            "Acne", "Actinic_Keratosis", "Benign_tumors", "Bullous", "Candidiasis",
            "DrugEruption", "Eczema", "Infestations_Bites", "Lichen", "Lupus",
            "Moles", "Psoriasis", "Rosacea", "Seborrh_Keratoses", "SkinCancer",
            "Sun_Sunlight_Damage", "Tinea", "Unknown_Normal", "Vascular_Tumors",
            "Vasculitis", "Vitiligo", "Warts",
        ],

        "out_json": "calibration_results.json",
    }

    main(CONFIG)
