# ============================================================
# DeepMediScan - generate_figures.py
# ------------------------------------------------------------
# Regenerates ALL Section 7.6 report figures (Figures 9-20) on the
# NEW leakage-safe checkpoints + leakage-safe Test set, so the figure
# images match the leakage-safe numbers already in the report text.
#
# For EACH of the four models it produces three PNGs:
#   - <model>_training_history.png   (loss / accuracy / AUC over epochs)
#       Figures 9, 12, 15, 18
#   - <model>_per_class_auc.png      (one-vs-rest Test AUC per class)
#       Figures 10, 13, 16, 19
#   - <model>_confusion_matrix.png   (Test set, counts + row-normalized %)
#       Figures 11, 14, 17, 20
#
# Model building + checkpoint loading is COPIED from main.py (same as
# full_metrics_eval.py), so the same checkpoints load unchanged and the
# class order is the verified ground truth.
#
# USAGE:
#   1. Confirm paths in the CONFIG block at the BOTTOM.
#   2. python generate_figures.py
#   3. PNGs are written to ./report_figures/ . Replace the old images in
#      the Word document (Section 7.6) with these.
#
# Requires: torch, timm, albumentations, pillow, scikit-learn,
#           matplotlib, numpy   (matplotlib is the only new dependency
#           vs. the eval scripts:  pip install matplotlib)
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

from sklearn.metrics import roc_auc_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ------------------------------------------------------------
# Report colour scheme (navy / orange, matches the deck + logo)
# ------------------------------------------------------------
NAVY   = "#1E2761"
ORANGE = "#E8742C"
GREEN  = "#2C7A4B"
GREY   = "#666666"
LIGHT  = "#CADCFC"


def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. MODEL  (copied verbatim from main.py)
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


def get_transform(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def load_image_tensor(path, img_size, device):
    np_img = np.array(Image.open(path).convert("RGB"))
    return get_transform(img_size)(image=np_img)["image"].unsqueeze(0).to(device)


def resolve_checkpoint_path(cands):
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None


def load_model_and_classes(model_name, ckpt_path, device, fallback_classes):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_classes = ckpt.get("num_classes", len(fallback_classes))
    model = SkinModel(model_name, n_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    c2i = ckpt.get("class_to_idx", None)
    if c2i and len(c2i) == n_classes:
        classes = [k for k, _ in sorted(c2i.items(), key=lambda kv: kv[1])]
    else:
        classes = list(fallback_classes[:n_classes])
    return model, classes, n_classes


# ============================================================
# 2. DATASET SCAN + INFERENCE
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
# 3. PLOTTING
# ============================================================
DISPLAY = {
    "Seborrh_Keratoses": "Seborrheic Keratoses",
    "Infestations_Bites": "Infestations & Bites",
    "Sun_Sunlight_Damage": "Sun/Sunlight Damage",
    "Unknown_Normal": "Unknown/Normal",
    "Vascular_Tumors": "Vascular Tumors",
    "Actinic_Keratosis": "Actinic Keratosis",
    "Benign_tumors": "Benign Tumors",
    "DrugEruption": "Drug Eruption",
    "SkinCancer": "Skin Cancer",
}


def disp(c):
    return DISPLAY.get(c, c)


def plot_training_history(history_json, model_title, out_png):
    with open(history_json) as f:
        H = json.load(f)
    hist = H["history"]
    best_epoch = H.get("best_epoch")
    ep   = [h["epoch"] for h in hist]
    tl   = [h["train_loss"] for h in hist]
    vl   = [h["val_loss"] for h in hist]
    ta   = [h["train_acc"] for h in hist]
    va   = [h["val_acc"] for h in hist]
    au   = [h["val_auc"] for h in hist]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(ep, tl, color=NAVY, label="Train Loss", linewidth=2)
    axes[0].plot(ep, vl, color=ORANGE, label="Val Loss", linewidth=2)
    axes[0].set_title("Loss", fontweight="bold"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, [x*100 for x in ta], color=NAVY, label="Train Acc", linewidth=2)
    axes[1].plot(ep, [x*100 for x in va], color=ORANGE, label="Val Acc", linewidth=2)
    axes[1].set_title("Accuracy (%)", fontweight="bold"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(ep, au, color=GREEN, label="Val AUC-ROC", linewidth=2)
    if best_epoch:
        be_auc = next((h["val_auc"] for h in hist if h["epoch"] == best_epoch), None)
        if be_auc is not None:
            axes[2].axvline(best_epoch, color=ORANGE, linestyle="--", alpha=0.7)
            axes[2].scatter([best_epoch], [be_auc], color=ORANGE, zorder=5,
                            label=f"Best (ep {best_epoch}, {be_auc:.4f})")
    axes[2].set_title("Validation AUC-ROC", fontweight="bold"); axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.suptitle(f"{model_title} — Training History (leakage-safe split)",
                 fontsize=13, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved {out_png}")


def plot_per_class_auc(y_true, probs, classes, model_title, out_png):
    n = len(classes)
    present = np.unique(y_true)
    aucs = []
    for c in range(n):
        if c not in present:
            aucs.append(np.nan); continue
        try:
            aucs.append(roc_auc_score((y_true == c).astype(int), probs[:, c]))
        except ValueError:
            aucs.append(np.nan)
    mean_auc = np.nanmean(aucs)

    order = np.argsort([a if not np.isnan(a) else 0 for a in aucs])
    names = [disp(classes[i]) for i in order]
    vals  = [aucs[i] for i in order]
    colors = [ORANGE if (not np.isnan(v) and v < 0.95) else NAVY for v in vals]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(range(n), [0 if np.isnan(v) else v for v in vals], color=colors)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0.80, 1.0)
    ax.axvline(0.90, color=GREY, linestyle="--", alpha=0.6, label="0.90 target")
    ax.axvline(mean_auc, color=GREEN, linestyle="-", alpha=0.7,
               label=f"Mean AUC = {mean_auc:.4f}")
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(min(v + 0.002, 0.998), i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("One-vs-Rest AUC-ROC (Test Set)")
    ax.set_title(f"{model_title} — Per-Class AUC-ROC (leakage-safe Test, Mean = {mean_auc:.4f})",
                 fontsize=12, fontweight="bold", color=NAVY)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved {out_png}  (mean AUC {mean_auc:.4f})")
    return mean_auc


def plot_confusion(y_true, probs, classes, model_title, out_png):
    y_pred = probs.argmax(axis=1)
    n = len(classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_pct = cm / cm.sum(axis=1, keepdims=True)
    cm_pct = np.nan_to_num(cm_pct)

    names = [disp(c) for c in classes]
    cmap = LinearSegmentedColormap.from_list("navymap", ["#FFFFFF", LIGHT, NAVY])

    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        fontsize=6, color="white" if cm_pct[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized fraction")
    ax.set_title(f"{model_title} — Confusion Matrix (counts; shaded by row %), leakage-safe Test",
                 fontsize=12, fontweight="bold", color=NAVY)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved {out_png}")


# ============================================================
# 4. MAIN
# ============================================================
def main(cfg):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    print(f"Device: {device}\nOutput: {cfg['out_dir']}")

    test_dir = cfg["test_dir"]
    summary = {}

    for model_name, meta in cfg["models"].items():
        ckpt = resolve_checkpoint_path(meta["ckpt"])
        title = meta["title"]
        print(f"\n=== {title} ({model_name}) ===")

        # 1) Training-history plot (from history JSON, no model needed)
        hist_json = meta["history"]
        if hist_json and os.path.exists(hist_json):
            plot_training_history(
                hist_json, title,
                os.path.join(cfg["out_dir"], f"{model_name}_training_history.png"))
        else:
            print(f"   (history JSON not found: {hist_json} — skipping history plot)")

        # 2) + 3) Per-class AUC + confusion matrix (need inference)
        if ckpt is None:
            print(f"   checkpoint not found — skipping AUC/confusion plots")
            continue
        model, classes, n_classes = load_model_and_classes(
            model_name, ckpt, device, cfg["fallback_classes"])
        img_size = meta.get("img_size", 224)

        paths, y_true = scan_split(test_dir, classes)
        print(f"   Test images: {len(paths)} — running inference...")
        probs = predict_probs(model, paths, img_size, device)

        mean_auc = plot_per_class_auc(
            y_true, probs, classes, title,
            os.path.join(cfg["out_dir"], f"{model_name}_per_class_auc.png"))
        plot_confusion(
            y_true, probs, classes, title,
            os.path.join(cfg["out_dir"], f"{model_name}_confusion_matrix.png"))
        summary[model_name] = {"mean_test_auc": round(float(mean_auc), 4),
                               "n_test": int(len(paths))}

    print("\n==== SUMMARY (mean Test AUC per model) ====")
    for m, d in summary.items():
        print(f"  {m:24} mean Test AUC = {d['mean_test_auc']}  (n={d['n_test']})")
    print(f"\nAll figures saved to: {cfg['out_dir']}")
    print("Replace the Section 7.6 images in the Word report with these PNGs.")


# ============================================================
# 5. CONFIG  ——  EDIT PATHS, THEN: python generate_figures.py
# ============================================================
if __name__ == "__main__":
    DERMAI_DIR    = r"C:\Users\M Shahid Asghar\Disease Dectection Project\DermAI"
    RETRAINED_DIR = os.path.join(DERMAI_DIR, "retrained_models")
    # history_*.json files (from training). Adjust folder if they live elsewhere.
    HISTORY_DIR   = DERMAI_DIR

    CONFIG = {
        "seed": 42,
        "test_dir": os.path.join(DERMAI_DIR, "SkinDisease_dataset_resplit", "resplit_output", "Test"),
        "out_dir":  os.path.join(DERMAI_DIR, "report_figures"),

        "models": {
            "efficientnet_b0": {
                "title": "EfficientNet-B0",
                "ckpt":  [os.path.join(RETRAINED_DIR, "best_efficientnet_b0_22class.pth")],
                "history": os.path.join(HISTORY_DIR, "history_efficientnet_b0.json"),
                "img_size": 224,
            },
            "efficientnet_b4": {
                "title": "EfficientNet-B4",
                "ckpt":  [os.path.join(RETRAINED_DIR, "best_efficientnet_b4_22class.pth")],
                "history": os.path.join(HISTORY_DIR, "history_efficientnet_b4.json"),
                "img_size": 224,
            },
            "resnet50": {
                "title": "ResNet-50",
                "ckpt":  [os.path.join(RETRAINED_DIR, "best_resnet50_22class.pth")],
                "history": os.path.join(HISTORY_DIR, "history_resnet50.json"),
                "img_size": 224,
            },
            "mobilenetv3_large_100": {
                "title": "MobileNetV3-Large",
                "ckpt":  [os.path.join(RETRAINED_DIR, "best_mobilenetv3_large_100_22class.pth"),
                          os.path.join(RETRAINED_DIR, "best_mobilenetv3_22class.pth")],
                "history": os.path.join(HISTORY_DIR, "history_mobilenetv3_large_100.json"),
                "img_size": 224,
            },
        },

        "fallback_classes": [
            "Acne", "Actinic_Keratosis", "Benign_tumors", "Bullous", "Candidiasis",
            "DrugEruption", "Eczema", "Infestations_Bites", "Lichen", "Lupus",
            "Moles", "Psoriasis", "Rosacea", "Seborrh_Keratoses", "SkinCancer",
            "Sun_Sunlight_Damage", "Tinea", "Unknown_Normal", "Vascular_Tumors",
            "Vasculitis", "Vitiligo", "Warts",
        ],
    }

    main(CONFIG)
