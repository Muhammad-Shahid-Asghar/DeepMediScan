# ============================================================
# DeepMediScan - full_metrics_eval.py
# ------------------------------------------------------------
# Authoritative per-model evaluation for the FYP report
# (Section 7.2 per-class, Section 7.4 full metrics table).
#
# For EACH model (EfficientNet-B0/B4, ResNet-50, MobileNetV3-Large)
# and EACH split (Val, Test) this script reports:
#   - Accuracy
#   - Macro-averaged F1
#   - Weighted-averaged F1
#   - Mean AUC-ROC (one-vs-rest, macro)
#   - Bootstrap 95% CI (1,000 resamples) for Accuracy and Macro-F1
#   - Full per-class Precision / Recall / F1 / support
#   - Per-class one-vs-rest AUC
#   - Confusion matrix (counts) + normalized (row %) confusion matrix
#
# Model building + checkpoint loading is COPIED EXACTLY from main.py
# (SkinModel class, class_to_idx derivation, Albumentations transform)
# so the same checkpoints load with zero changes and the class order
# is the verified ground truth.
#
# Reproducible: fixed SEED (default 42). Bootstrap uses its own seeded RNG.
#
# USAGE:
#   1. Confirm the paths in the CONFIG block at the BOTTOM of this file.
#   2. python full_metrics_eval.py
#   3. Copy the printed tables into the chat so the report can be updated.
#      A machine-readable copy is also written to:  full_metrics_results.json
# ============================================================

import os
import io
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import timm

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

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
        # FIX: dummy forward pass to get ACTUAL output features
        # backbone.num_features is wrong for MobileNetV3 (960 vs actual 1280)
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
    pil_img   = Image.open(path).convert("RGB")
    np_img    = np.array(pil_img)
    transform = get_transform(img_size)
    return transform(image=np_img)["image"].unsqueeze(0).to(device)


# ============================================================
# 3. CHECKPOINT LOADING  (mirrors main.py logic)
# ============================================================
def resolve_checkpoint_path(model_name, candidate_paths):
    """Return the first existing path from a list of candidates."""
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

    # Derive class order from class_to_idx (primary), exactly like main.py
    ckpt_class_to_idx = checkpoint.get("class_to_idx", None)
    ckpt_classes      = checkpoint.get("class_names", None)
    if ckpt_class_to_idx and len(ckpt_class_to_idx) == n_classes:
        ordered = sorted(ckpt_class_to_idx.items(), key=lambda kv: kv[1])
        classes = [name for name, idx in ordered]
        src = "class_to_idx (verified)"
    elif ckpt_classes and len(ckpt_classes) == n_classes:
        classes = list(ckpt_classes)
        src = "class_names"
    else:
        classes = list(fallback_classes[:n_classes])
        src = "FALLBACK hardcoded list (VERIFY!)"

    info = {
        "ckpt_auc":   round(float(checkpoint.get("auc", 0) or 0), 4),
        "ckpt_epoch": checkpoint.get("epoch", "?"),
        "class_src":  src,
    }
    return model, classes, n_classes, info


# ============================================================
# 4. DATASET SCAN  (ImageFolder-style, alphabetical = index order)
# ============================================================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def scan_split(split_dir, class_order):
    """
    Walk split_dir/<ClassName>/*.img and return (paths, y_true_idx).
    class_order defines the index mapping (must match the model's class order).
    Folder names must match the class names exactly.
    """
    name_to_idx = {c: i for i, c in enumerate(class_order)}
    paths, labels = [], []
    missing_folders = []

    for cname in class_order:
        cdir = os.path.join(split_dir, cname)
        if not os.path.isdir(cdir):
            missing_folders.append(cname)
            continue
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(cdir, fn))
                labels.append(name_to_idx[cname])

    # Also warn about any folder present on disk but not in class_order
    on_disk = {d for d in os.listdir(split_dir)
               if os.path.isdir(os.path.join(split_dir, d))}
    extra_folders = sorted(on_disk - set(class_order))

    return paths, np.array(labels, dtype=np.int64), missing_folders, extra_folders


# ============================================================
# 5. INFERENCE  ->  probability matrix
# ============================================================
@torch.no_grad()
def predict_probs(model, paths, img_size, device, batch_log_every=500):
    """Return an (N, C) softmax-probability matrix, one row per image."""
    rows = []
    for i, p in enumerate(paths):
        tensor = load_image_tensor(p, img_size, device)
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        rows.append(probs)
        if batch_log_every and (i + 1) % batch_log_every == 0:
            print(f"      ... {i + 1}/{len(paths)} images", flush=True)
    return np.vstack(rows)


# ============================================================
# 6. METRICS  (+ bootstrap CIs)
# ============================================================
def safe_macro_auc(y_true, probs, n_classes):
    """
    Macro one-vs-rest AUC. Classes absent from y_true are skipped
    (roc_auc_score can't score a class with no positive samples).
    Returns (mean_auc, per_class_auc_dict_by_index).
    """
    per_class = {}
    present = np.unique(y_true)
    for c in range(n_classes):
        if c not in present:
            per_class[c] = float("nan")
            continue
        y_bin = (y_true == c).astype(int)
        try:
            per_class[c] = roc_auc_score(y_bin, probs[:, c])
        except ValueError:
            per_class[c] = float("nan")
    vals = [v for v in per_class.values() if not np.isnan(v)]
    mean_auc = float(np.mean(vals)) if vals else float("nan")
    return mean_auc, per_class


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42):
    """
    Bootstrap 95% CI for Accuracy and Macro-F1 (percentile method).
    Resamples image indices with replacement n_boot times.
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    accs, f1s = [], []
    idx_all = np.arange(n)
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
    acc_ci = (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))
    f1_ci  = (float(np.percentile(f1s, 2.5)),  float(np.percentile(f1s, 97.5)))
    return acc_ci, f1_ci


def evaluate_split(model_name, classes, n_classes, probs, y_true,
                   n_boot, seed):
    y_pred = probs.argmax(axis=1)

    acc        = accuracy_score(y_true, y_pred)
    macro_f1   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1= f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mean_auc, per_class_auc = safe_macro_auc(y_true, probs, n_classes)

    acc_ci, f1_ci = bootstrap_ci(y_true, y_pred, n_boot=n_boot, seed=seed)

    # Per-class precision/recall/f1/support
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_row = cm / cm.sum(axis=1, keepdims=True)
    cm_row = np.nan_to_num(cm_row)

    return {
        "accuracy":     float(acc),
        "macro_f1":     float(macro_f1),
        "weighted_f1":  float(weighted_f1),
        "mean_auc":     float(mean_auc),
        "acc_ci":       acc_ci,
        "macro_f1_ci":  f1_ci,
        "per_class": {
            classes[i]: {
                "precision": float(p[i]),
                "recall":    float(r[i]),
                "f1":        float(f[i]),
                "support":   int(s[i]),
                "auc":       (None if np.isnan(per_class_auc[i])
                              else float(per_class_auc[i])),
                "diag_pct":  (float(cm_row[i, i]) if s[i] > 0 else None),
            }
            for i in range(n_classes)
        },
        "n_images": int(len(y_true)),
    }


# ============================================================
# 7. PRINTING
# ============================================================
def pct(x):
    return "  N/A " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:6.2f}%"


def f4(x):
    return " N/A  " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def print_summary_table(all_results):
    print("\n" + "=" * 110)
    print("TABLE 7.4 — FULL CLASSIFICATION METRICS (Authoritative)")
    print("=" * 110)
    header = (f"{'Model':<20} {'Split':<6} {'Accuracy':>9} {'Macro-F1':>9} "
              f"{'Weighted-F1':>12} {'Mean AUC':>9} "
              f"{'Acc 95% CI':>20} {'Macro-F1 95% CI':>22}")
    print(header)
    print("-" * 110)
    for model_name, splits in all_results.items():
        for split_name, m in splits.items():
            acc_ci = m["acc_ci"]; f1_ci = m["macro_f1_ci"]
            print(f"{model_name:<20} {split_name:<6} "
                  f"{m['accuracy']*100:8.2f}% "
                  f"{m['macro_f1']:9.4f} "
                  f"{m['weighted_f1']:12.4f} "
                  f"{m['mean_auc']:9.4f} "
                  f"[{acc_ci[0]*100:5.2f}%, {acc_ci[1]*100:5.2f}%] "
                  f"   [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
    print("-" * 110)


def print_per_class(model_name, split_name, m):
    print(f"\n--- Per-Class Metrics: {model_name} ({split_name}) ---")
    print(f"{'Class':<22} {'Prec':>7} {'Recall':>7} {'F1':>7} "
          f"{'AUC':>7} {'Diag%':>7} {'Support':>8}")
    print("-" * 70)
    for cname, d in m["per_class"].items():
        print(f"{cname:<22} "
              f"{d['precision']:7.3f} {d['recall']:7.3f} {d['f1']:7.3f} "
              f"{f4(d['auc']):>7} {pct(d['diag_pct']):>7} {d['support']:8d}")
    print("-" * 70)


# ============================================================
# 8. MAIN
# ============================================================
def main(cfg):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed:   {cfg['seed']}")
    print(f"Bootstrap resamples: {cfg['n_boot']}")

    all_results = {}

    for model_name, ckpt_candidates in cfg["model_files"].items():
        ckpt_path = resolve_checkpoint_path(model_name, ckpt_candidates)
        if ckpt_path is None:
            print(f"\nSKIP {model_name}: no checkpoint found in {ckpt_candidates}")
            continue

        print("\n" + "=" * 70)
        print(f"MODEL: {model_name}")
        print(f"  checkpoint: {ckpt_path}")
        model, classes, n_classes, info = load_model_and_classes(
            model_name, ckpt_path, device, cfg["fallback_classes"]
        )
        print(f"  classes ({n_classes}) source: {info['class_src']}")
        print(f"  checkpoint best Val AUC (stored): {info['ckpt_auc']}  "
              f"epoch: {info['ckpt_epoch']}")

        img_size = cfg["img_sizes"].get(model_name, 224)
        all_results[model_name] = {}

        for split_name, split_dir in cfg["splits"].items():
            if not os.path.isdir(split_dir):
                print(f"  [{split_name}] folder not found: {split_dir} — skipping")
                continue
            paths, y_true, missing, extra = scan_split(split_dir, classes)
            if missing:
                print(f"  [{split_name}] WARNING missing class folders: {missing}")
            if extra:
                print(f"  [{split_name}] WARNING folders on disk not in class list "
                      f"(ignored): {extra}")
            print(f"  [{split_name}] {len(paths)} images across "
                  f"{len(set(y_true.tolist()))} classes — running inference...")

            probs = predict_probs(model, paths, img_size, device)
            res   = evaluate_split(model_name, classes, n_classes,
                                   probs, y_true, cfg["n_boot"], cfg["seed"])
            all_results[model_name][split_name] = res

            print(f"  [{split_name}] Acc={res['accuracy']*100:.2f}%  "
                  f"MacroF1={res['macro_f1']:.4f}  "
                  f"WeightedF1={res['weighted_f1']:.4f}  "
                  f"MeanAUC={res['mean_auc']:.4f}")

    # ---- Reports ----
    print_summary_table(all_results)

    # Per-class (Test only by default; set cfg['per_class_splits'] to change)
    for model_name, splits in all_results.items():
        for split_name in cfg["per_class_splits"]:
            if split_name in splits:
                print_per_class(model_name, split_name, splits[split_name])

    # ---- Save machine-readable copy ----
    out = cfg.get("out_json", "full_metrics_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved machine-readable results to: {out}")
    print("\nDONE. Paste the TABLE 7.4 block (and per-class tables) into the chat.")


# ============================================================
# 9. CONFIG  ——  EDIT THESE PATHS, THEN: python full_metrics_eval.py
# ============================================================
if __name__ == "__main__":
    # Root of your project (same DERMAI_DIR as main.py)
    DERMAI_DIR = r"C:\Users\M Shahid Asghar\Disease Dectection Project\DermAI"

    # NEW leakage-safe checkpoints live in retrained_models/.
    # Each model lists CANDIDATE paths in priority order — the first that
    # exists is used. This handles the MobileNetV3 filename difference
    # (training saved best_mobilenetv3_large_100_22class.pth, while the old
    # main.py expected best_mobilenetv3_22class.pth).
    RETRAINED_DIR = os.path.join(DERMAI_DIR, "retrained_models")

    CONFIG = {
        "seed":    42,
        "n_boot":  1000,            # bootstrap resamples for 95% CI

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

        # NEW leakage-safe split folders (ImageFolder layout: <split>/<Class>/*.jpg)
        "splits": {
            "Val":  os.path.join(DERMAI_DIR, "SkinDisease_dataset_resplit", "resplit_output", "Val"),
            "Test": os.path.join(DERMAI_DIR, "SkinDisease_dataset_resplit", "resplit_output", "Test"),
        },

        # Which splits to print full per-class tables for
        "per_class_splits": ["Test"],

        # Last-resort class order if a checkpoint somehow lacks class_to_idx.
        # This is the verified 22-class ground-truth order (matches main.py).
        "fallback_classes": [
            "Acne", "Actinic_Keratosis", "Benign_tumors", "Bullous", "Candidiasis",
            "DrugEruption", "Eczema", "Infestations_Bites", "Lichen", "Lupus",
            "Moles", "Psoriasis", "Rosacea", "Seborrh_Keratoses", "SkinCancer",
            "Sun_Sunlight_Damage", "Tinea", "Unknown_Normal", "Vascular_Tumors",
            "Vasculitis", "Vitiligo", "Warts",
        ],

        "out_json": "full_metrics_results.json",
    }

    main(CONFIG)
