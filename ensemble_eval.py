# ============================================================
# DeepMediScan - ensemble_eval.py
# ------------------------------------------------------------
# Equal-weight ensemble evaluation for the FYP report (Section 7.5).
#
# Loads all four retrained models, runs each on the leakage-safe Test set,
# averages their softmax probabilities with EQUAL weight (0.25 each) —
# the exact configuration deployed in main.py's /predict/ensemble endpoint —
# and reports:
#   - Per-model Test Accuracy / Macro-F1 / Mean AUC / mean latency-per-image
#   - Ensemble Test Accuracy / Macro-F1 / Mean AUC
#   - Accuracy improvement of ensemble over the best single model
#   - McNemar significance test (ensemble vs. best single model) with p-value
#
# Model building + checkpoint loading is copied from main.py so the same
# checkpoints load unchanged and the class order is the verified ground truth.
#
# Reproducible: fixed SEED (default 42).
#
# USAGE:
#   1. Confirm the paths in the CONFIG block at the BOTTOM.
#   2. python ensemble_eval.py
#   3. Paste the printed ENSEMBLE block into the chat.
#      Machine-readable copy: ensemble_results.json
#
# NOTE: run full_metrics_eval.py as well — that produces the per-model
# table with confidence intervals (Section 7.4). This script focuses on
# the ensemble comparison (Section 7.5).
# ============================================================

import os
import json
import time
import numpy as np

import torch
import torch.nn as nn
import timm

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


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
    pil_img   = Image.open(path).convert("RGB")
    np_img    = np.array(pil_img)
    transform = get_transform(img_size)
    return transform(image=np_img)["image"].unsqueeze(0).to(device)


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

    return model, classes, n_classes, src


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
# 5. INFERENCE  ->  per-model probability matrix (+ latency)
# ============================================================
@torch.no_grad()
def predict_probs_with_latency(model, paths, img_size, device, log_every=500):
    """
    Returns (probs[N,C], mean_latency_ms_per_image).
    Latency = pure forward-pass time (preprocessing excluded), averaged
    over all images, matching how main.py serves a single inference.
    """
    rows = []
    total_fwd = 0.0
    use_cuda = device.type == "cuda"
    for i, p in enumerate(paths):
        tensor = load_image_tensor(p, img_size, device)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(tensor)
        if use_cuda:
            torch.cuda.synchronize()
        total_fwd += (time.perf_counter() - t0)
        rows.append(torch.softmax(logits, dim=1)[0].cpu().numpy())
        if log_every and (i + 1) % log_every == 0:
            print(f"      ... {i + 1}/{len(paths)}", flush=True)
    mean_latency_ms = (total_fwd / max(len(paths), 1)) * 1000.0
    return np.vstack(rows), mean_latency_ms


# ============================================================
# 6. METRICS
# ============================================================
def macro_auc(y_true, probs, n_classes):
    present = np.unique(y_true)
    vals = []
    for c in range(n_classes):
        if c not in present:
            continue
        y_bin = (y_true == c).astype(int)
        try:
            vals.append(roc_auc_score(y_bin, probs[:, c]))
        except ValueError:
            pass
    return float(np.mean(vals)) if vals else float("nan")


def basic_metrics(y_true, probs, n_classes):
    y_pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mean_auc": float(macro_auc(y_true, probs, n_classes)),
    }


def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar test comparing two classifiers on the SAME samples.
    pred_a = best single model, pred_b = ensemble.
    b = a correct & b wrong ; c = a wrong & b correct.
    Uses exact binomial when (b+c) is small, else chi-square with
    continuity correction. Returns (b, c, statistic_or_None, p_value).
    """
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)
    b = int(np.sum(a_correct & ~b_correct))   # only A right
    c = int(np.sum(~a_correct & b_correct))   # only B right
    n = b + c

    if n == 0:
        return b, c, None, 1.0

    # Exact binomial (two-sided) for small discordant counts
    if n < 25:
        from math import comb
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        p = min(1.0, 2.0 * tail)
        return b, c, None, float(p)

    # Chi-square with continuity correction
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    try:
        from scipy.stats import chi2
        p = float(chi2.sf(stat, df=1))
    except Exception:
        # Fallback normal approximation if scipy unavailable
        from math import erfc, sqrt
        z = (abs(b - c) - 1) / sqrt(b + c)
        p = float(erfc(z / sqrt(2)))
    return b, c, float(stat), p


# ============================================================
# 7. MAIN
# ============================================================
def main(cfg):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed:   {cfg['seed']}")
    print(f"Ensemble weights: {cfg['ensemble_weights']}")

    split_name = cfg["eval_split_name"]
    split_dir  = cfg["eval_split_dir"]
    if not os.path.isdir(split_dir):
        raise SystemExit(f"Test split folder not found: {split_dir}")

    # Load models
    models, model_classes, model_imgsize = {}, {}, {}
    reference_classes = None
    for model_name, cands in cfg["model_files"].items():
        ckpt = resolve_checkpoint_path(cands)
        if ckpt is None:
            print(f"SKIP {model_name}: no checkpoint found in {cands}")
            continue
        model, classes, n_classes, src = load_model_and_classes(
            model_name, ckpt, device, cfg["fallback_classes"]
        )
        models[model_name]        = model
        model_classes[model_name] = classes
        model_imgsize[model_name] = cfg["img_sizes"].get(model_name, 224)
        print(f"Loaded {model_name:<24} classes={n_classes} ({src})  ckpt={os.path.basename(ckpt)}")
        if reference_classes is None:
            reference_classes = classes
        elif classes != reference_classes:
            print(f"  WARNING: {model_name} class order differs from reference! "
                  f"Ensemble averaging assumes identical class order.")

    if len(models) < 2:
        raise SystemExit("Need at least 2 models loaded to evaluate an ensemble.")

    n_classes = len(reference_classes)

    # Scan the Test split once (labels indexed by reference class order)
    paths, y_true = scan_split(split_dir, reference_classes)
    print(f"\n[{split_name}] {len(paths)} images, {n_classes} classes")

    # Per-model inference
    per_model_probs   = {}
    per_model_metrics = {}
    per_model_latency = {}
    for model_name, model in models.items():
        print(f"\nInferring with {model_name} ...")
        probs, lat_ms = predict_probs_with_latency(
            model, paths, model_imgsize[model_name], device
        )
        per_model_probs[model_name]   = probs
        per_model_metrics[model_name] = basic_metrics(y_true, probs, n_classes)
        per_model_latency[model_name] = lat_ms
        m = per_model_metrics[model_name]
        print(f"  Acc={m['accuracy']*100:.2f}%  MacroF1={m['macro_f1']:.4f}  "
              f"MeanAUC={m['mean_auc']:.4f}  Latency={lat_ms:.1f} ms/img")

    # Equal-weight ensemble
    weights = cfg["ensemble_weights"]
    ens = None
    total_w = 0.0
    for model_name, probs in per_model_probs.items():
        w = weights.get(model_name, 1.0 / len(per_model_probs))
        ens = (w * probs) if ens is None else (ens + w * probs)
        total_w += w
    ens = ens / total_w
    ens_metrics = basic_metrics(y_true, ens, n_classes)
    ens_latency = sum(per_model_latency.values())  # sequential

    # Best single model (by Accuracy) for the comparison
    best_single = max(per_model_metrics,
                      key=lambda k: per_model_metrics[k]["accuracy"])
    best_pred   = per_model_probs[best_single].argmax(axis=1)
    ens_pred    = ens.argmax(axis=1)
    b, c, stat, p = mcnemar_test(y_true, best_pred, ens_pred)

    acc_gain = (ens_metrics["accuracy"] - per_model_metrics[best_single]["accuracy"]) * 100

    # ---- Report ----
    print("\n" + "=" * 92)
    print("TABLE 7.5 — ENSEMBLE EVALUATION (equal-weight avg of all models)")
    print("=" * 92)
    print(f"{'Model':<30} {'Accuracy':>9} {'Macro-F1':>9} {'Mean AUC':>9} {'Latency/img':>13}")
    print("-" * 92)
    for model_name in per_model_metrics:
        m = per_model_metrics[model_name]
        print(f"{model_name:<30} {m['accuracy']*100:8.2f}% {m['macro_f1']:9.4f} "
              f"{m['mean_auc']:9.4f} {per_model_latency[model_name]:10.1f} ms")
    print(f"{'Ensemble (all, equal-weight)':<30} "
          f"{ens_metrics['accuracy']*100:8.2f}% {ens_metrics['macro_f1']:9.4f} "
          f"{ens_metrics['mean_auc']:9.4f} {ens_latency:8.1f} ms (seq)")
    print("-" * 92)
    print(f"\nBest single model (by Accuracy): {best_single} "
          f"({per_model_metrics[best_single]['accuracy']*100:.2f}%)")
    print(f"Ensemble accuracy gain over best single model: {acc_gain:+.2f} points")
    print(f"\nMcNemar test  (best single '{best_single}'  vs  ensemble):")
    print(f"  only-best-correct (b) = {b}")
    print(f"  only-ensemble-correct (c) = {c}")
    if stat is not None:
        print(f"  chi-square statistic (cont. corrected) = {stat:.4f}")
    else:
        print(f"  exact binomial test used (small discordant count)")
    sig = "STATISTICALLY SIGNIFICANT" if p < 0.05 else "NOT significant"
    print(f"  p-value = {p:.6g}   ->  {sig} at alpha=0.05")
    print("=" * 92)

    # ---- Save ----
    results = {
        "split": split_name,
        "n_images": int(len(y_true)),
        "ensemble_weights": weights,
        "per_model": {
            mn: {**per_model_metrics[mn], "latency_ms": per_model_latency[mn]}
            for mn in per_model_metrics
        },
        "ensemble": {**ens_metrics, "latency_ms_sequential": ens_latency},
        "best_single_model": best_single,
        "accuracy_gain_points": acc_gain,
        "mcnemar": {"b_only_best": b, "c_only_ensemble": c,
                    "statistic": stat, "p_value": p},
    }
    out = cfg.get("out_json", "ensemble_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved machine-readable results to: {out}")
    print("\nDONE. Paste the TABLE 7.5 block into the chat.")


# ============================================================
# 8. CONFIG  ——  EDIT THESE PATHS, THEN: python ensemble_eval.py
# ============================================================
if __name__ == "__main__":
    DERMAI_DIR    = r"C:\Users\M Shahid Asghar\Disease Dectection Project\DermAI"
    RETRAINED_DIR = os.path.join(DERMAI_DIR, "retrained_models")

    CONFIG = {
        "seed": 42,

        # Equal weight (0.25 each) — the deployed, evaluated configuration.
        "ensemble_weights": {
            "efficientnet_b0":       0.25,
            "efficientnet_b4":       0.25,
            "resnet50":              0.25,
            "mobilenetv3_large_100": 0.25,
        },

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

        # Evaluate the ensemble on the leakage-safe Test set
        "eval_split_name": "Test",
        "eval_split_dir":  os.path.join(DERMAI_DIR, "SkinDisease_dataset_resplit", "resplit_output", "Test"),

        "fallback_classes": [
            "Acne", "Actinic_Keratosis", "Benign_tumors", "Bullous", "Candidiasis",
            "DrugEruption", "Eczema", "Infestations_Bites", "Lichen", "Lupus",
            "Moles", "Psoriasis", "Rosacea", "Seborrh_Keratoses", "SkinCancer",
            "Sun_Sunlight_Damage", "Tinea", "Unknown_Normal", "Vascular_Tumors",
            "Vasculitis", "Vitiligo", "Warts",
        ],

        "out_json": "ensemble_results.json",
    }

    main(CONFIG)
