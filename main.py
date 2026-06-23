# ============================================================
# DeepMediScan - FastAPI Backend
# Multi-Model Support: EfficientNet-B0/B4, ResNet50, MobileNetV3
# + Ensemble Mode
# 22 Classes (21 Skin Diseases + Normal Skin)
# ============================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
import io
import os
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv

# Load secrets from .env file (must be in same folder as main.py)
load_dotenv()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# ============================================================
# 1. CONFIG
# ============================================================
# Project directory.
# By default this is the folder where main.py lives, so the app works on any
# computer without editing this line. If your models/dataset live somewhere
# else, set the DERMAI_DIR environment variable (e.g. in your .env file) and
# it will be used instead.
DERMAI_DIR  = os.environ.get("DERMAI_DIR", os.path.dirname(os.path.abspath(__file__)))
NUM_CLASSES = 22   # 21 skin diseases + 1 Normal Skin
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Project / Academic Information ──
PROJECT_INFO = {
    "project_title" : "DeepMediScan — A Multi-Model Deep Learning Prototype for 22-Class Dermoscopic Skin Image Classification",
    "developed_by"  : "Muhammad Shahid Asghar",
    "roll_no"       : "F22BDATS1M02033",
    "supervised_by" : "Dr. Akmal Khan",
    "department"    : "Department of Data Science",
    "university"    : "The Islamia University of Bahawalpur (IUB)",
    "session"       : "Fall 2022–2026",
    "province"      : "Punjab",
    "country"       : "Pakistan",
    "team"          : "Individual Project (No Team) — developed solely by Muhammad Shahid Asghar",
    "note"          : (
        "This project was developed as an individual effort by Muhammad Shahid Asghar "
        "(Roll No. F22BDATS1M02033), a student of the Department of Data Science, "
        "The Islamia University of Bahawalpur (IUB), Punjab, Pakistan, under the "
        "supervision of Dr. Akmal Khan."
    ),
    "disclaimer"    : (
        "DeepMediScan is a research prototype for screening support only. "
        "It has not undergone clinical validation and must never replace a doctor's diagnosis."
    ),
}

# ── Available Models ──
# Jo model train ho chuka ho uska path yahan set karo
# Updated to the leakage-safe RETRAINED checkpoints (Section 4.3 of the report).
# These are the models trained on the leakage-safe re-split and used for all
# final reported results. The old pre-resplit checkpoints in DERMAI_DIR are
# superseded and should NOT be served.
RETRAINED_DIR = os.path.join(DERMAI_DIR, "retrained_models")
MODEL_FILES = {
    "efficientnet_b0"       : os.path.join(RETRAINED_DIR, "best_efficientnet_b0_22class.pth"),
    "efficientnet_b4"       : os.path.join(RETRAINED_DIR, "best_efficientnet_b4_22class.pth"),
    "resnet50"              : os.path.join(RETRAINED_DIR, "best_resnet50_22class.pth"),
    "mobilenetv3_large_100" : os.path.join(RETRAINED_DIR, "best_mobilenetv3_large_100_22class.pth"),
}

# ── Ensemble Weights ──
# IMPORTANT: equal-weight (0.25 each) on purpose. This is the exact
# configuration that was formally evaluated on the leakage-safe Test set
# (Section 7.5 of the FYP report): 80.60% accuracy, 0.9786 mean AUC, a
# statistically significant +2.36-point improvement over the best single
# model (McNemar p = 0.0004). The previous custom weighting (20/25/25/30,
# favoring MobileNetV3-Large) was never evaluated — deploying it instead of
# the tested equal-weight scheme would mean shipping unverified behavior.
# If you want to try a custom weighting in the future, re-run
# ensemble_eval.py with the new weights and confirm it still helps before
# changing this.
ENSEMBLE_WEIGHTS = {
    "efficientnet_b0"       : 0.25,
    "efficientnet_b4"       : 0.25,
    "resnet50"              : 0.25,
    "mobilenetv3_large_100" : 0.25,
}

# ── Image sizes per model ──
MODEL_IMG_SIZES = {
    "efficientnet_b0"       : 224,
    "efficientnet_b4"       : 224,
    "resnet50"              : 224,
    "mobilenetv3_large_100" : 224,
}

# 22 Classes — same order as training scripts (alphabetical folder order)
# 22 Classes — RAW keys must exactly match each checkpoint's class_to_idx
# (this is the ground truth: alphabetical folder order used by ImageFolder
# during training). These raw keys are used internally for all dictionary
# lookups (severity, description, Urdu name). DISPLAY_NAME_MAP converts them
# to clean, human-readable names shown to the user.
CLASS_NAMES = [
    "Acne",
    "Actinic_Keratosis",
    "Benign_tumors",
    "Bullous",
    "Candidiasis",
    "DrugEruption",
    "Eczema",
    "Infestations_Bites",
    "Lichen",
    "Lupus",
    "Moles",
    "Psoriasis",
    "Rosacea",
    "Seborrh_Keratoses",
    "SkinCancer",
    "Sun_Sunlight_Damage",
    "Tinea",
    "Unknown_Normal",
    "Vascular_Tumors",
    "Vasculitis",
    "Vitiligo",
    "Warts",
]

# Raw checkpoint key -> clean display name shown to the user
DISPLAY_NAME_MAP = {
    "Acne"                : "Acne",
    "Actinic_Keratosis"   : "Actinic Keratosis",
    "Benign_tumors"       : "Benign Tumors",
    "Bullous"             : "Bullous Disorders",
    "Candidiasis"         : "Candidiasis",
    "DrugEruption"        : "Drug Eruption",
    "Eczema"              : "Eczema",
    "Infestations_Bites"  : "Infestations & Bites",
    "Lichen"              : "Lichen Planus",
    "Lupus"               : "Lupus",
    "Moles"               : "Moles (Melanocytic Nevi)",
    "Psoriasis"           : "Psoriasis",
    "Rosacea"             : "Rosacea",
    "Seborrh_Keratoses"   : "Seborrheic Keratoses",
    "SkinCancer"          : "Skin Cancer",
    "Sun_Sunlight_Damage" : "Sun / Sunlight Damage",
    "Tinea"               : "Tinea (Ringworm)",
    "Unknown_Normal"      : "Normal / Unknown Skin",
    "Vascular_Tumors"     : "Vascular Tumors",
    "Vasculitis"          : "Vasculitis",
    "Vitiligo"            : "Vitiligo",
    "Warts"               : "Warts",
}

URDU_NAMES = {
    "Acne"                : "مہاسے",
    "Actinic_Keratosis"   : "ایکٹینک کیراٹوسس",
    "Benign_tumors"       : "سومی رسولی",
    "Bullous"             : "آبلے دار جِلدی امراض",
    "Candidiasis"         : "کینڈیڈیاسس (فنگل انفیکشن)",
    "DrugEruption"        : "دوا کا جِلدی ری ایکشن",
    "Eczema"              : "ایگزیما",
    "Infestations_Bites"  : "جِلدی کیڑے اور کاٹنا",
    "Lichen"              : "لائیکن پلینس",
    "Lupus"               : "لیوپس",
    "Moles"               : "تل (میلانوسٹک نیوی)",
    "Psoriasis"           : "چنبل",
    "Rosacea"             : "روزاسیا",
    "Seborrh_Keratoses"   : "سیبوریک کیراٹوسس",
    "SkinCancer"          : "جلد کا کینسر",
    "Sun_Sunlight_Damage" : "دھوپ سے جِلدی نقصان",
    "Tinea"               : "داد / فنگل انفیکشن",
    "Unknown_Normal"      : "نارمل / غیر واضح جلد",
    "Vascular_Tumors"     : "عروقی رسولی",
    "Vasculitis"          : "ویسکولائٹس (خون کی نالیوں کی سوزش)",
    "Vitiligo"            : "برص (سفید داغ)",
    "Warts"               : "مسے",
}

SEVERITY_INFO = {
    "Acne"                : ("Low",       "#10b981", "Keep skin clean. Use prescribed topical treatments. Consult a dermatologist if severe or cystic."),
    "Actinic_Keratosis"   : ("Moderate",  "#f59e0b", "Pre-cancerous lesion. Consult a dermatologist for treatment — do not ignore."),
    "Benign_tumors"       : ("Low",       "#10b981", "Usually non-cancerous. Monitor for changes in size, shape, or color and consult a doctor if it changes."),
    "Bullous"             : ("HIGH RISK", "#ef4444", "Blistering skin disorders can be serious autoimmune conditions. Please consult a dermatologist promptly."),
    "Candidiasis"         : ("Low",       "#10b981", "Fungal/yeast infection. Treat with antifungal medication. Keep area clean and dry. See a doctor if it persists."),
    "DrugEruption"        : ("Moderate",  "#f59e0b", "Possible reaction to a medication. Identify and report the suspected drug to a doctor; do not stop prescribed medication without medical advice."),
    "Eczema"              : ("Moderate",  "#f59e0b", "Monitor regularly. Use prescribed moisturizers. Consult dermatologist if it worsens."),
    "Infestations_Bites"  : ("Moderate",  "#f59e0b", "Possible insect bite, mite, or parasitic infestation. Keep area clean; consult a doctor if it spreads or does not improve."),
    "Lichen"              : ("Moderate",  "#f59e0b", "Chronic inflammatory condition (Lichen Planus). Consult a dermatologist for a confirmed diagnosis and treatment plan."),
    "Lupus"               : ("HIGH RISK", "#ef4444", "URGENT: Lupus is a systemic autoimmune disease. Consult a specialist (rheumatologist/dermatologist) immediately."),
    "Moles"               : ("Low",       "#10b981", "Usually benign. Monitor for changes in size, shape, or color (the ABCDE rule). Annual skin check recommended."),
    "Psoriasis"           : ("Moderate",  "#f59e0b", "Chronic autoimmune condition. Use prescribed treatments. Consult a dermatologist for a management plan."),
    "Rosacea"             : ("Low",       "#10b981", "Avoid triggers (sun, spicy food, alcohol). Use prescribed topical or oral treatments."),
    "Seborrh_Keratoses"   : ("Low",       "#10b981", "Benign skin growth. No treatment needed unless irritated. Consult a doctor if unsure."),
    "SkinCancer"          : ("HIGH RISK", "#ef4444", "URGENT: This result suggests a possible malignant (cancerous) lesion. Please consult a dermatologist or oncologist immediately. Note: this model's accuracy on this specific class is currently lower than other classes (55–61% on the leakage-safe test set) — a low-confidence or negative result here should NEVER be treated as ruling out cancer. If you have any concern about a lesion, please get it checked by a doctor regardless of this result."),
    "Sun_Sunlight_Damage" : ("Moderate",  "#f59e0b", "Sun/sunlight-related skin damage. Use sun protection and monitor the area, as it can progress to pre-cancerous changes over time."),
    "Tinea"               : ("Moderate",  "#f59e0b", "Fungal infection (ringworm). Use antifungal creams. Keep area clean and dry. See a doctor if persistent."),
    "Unknown_Normal"      : ("Normal",    "#10b981", "No specific skin disease detected. Maintain good skincare routine and consult a dermatologist for regular checkups."),
    "Vascular_Tumors"     : ("Moderate",  "#f59e0b", "Blood-vessel-related growth. Usually benign but should be evaluated by a doctor, especially if growing or bleeding."),
    "Vasculitis"          : ("HIGH RISK", "#ef4444", "Inflammation of blood vessels — can be a sign of a systemic condition. Please consult a doctor promptly for evaluation."),
    "Vitiligo"            : ("Low",       "#10b981", "Loss of skin pigment. Not dangerous, but consult a dermatologist for management and monitoring options."),
    "Warts"               : ("Low",       "#10b981", "Usually harmless viral skin growths. Can be treated or left to resolve naturally. Consult a doctor if it spreads or is bothersome."),
}

DESCRIPTIONS = {
    "Acne"                : "Acne is a common skin condition caused by clogged hair follicles with oil and dead skin cells, leading to pimples, blackheads, and cysts.",
    "Actinic_Keratosis"   : "A rough, scaly patch on skin caused by years of sun exposure. Considered pre-cancerous and should be treated to prevent progression.",
    "Benign_tumors"       : "Non-cancerous skin growths or tumors. Generally harmless but should be monitored for any changes.",
    "Bullous"             : "A category of skin disorders characterized by fluid-filled blisters, sometimes linked to autoimmune conditions (e.g. pemphigus, bullous pemphigoid).",
    "Candidiasis"         : "A fungal (yeast) skin infection caused by Candida, often appearing as red, itchy, sometimes moist patches, especially in skin folds.",
    "DrugEruption"        : "A skin reaction triggered by a medication, ranging from mild rashes to more serious reactions. Identifying the causative drug is important.",
    "Eczema"              : "Eczema is a chronic inflammatory skin condition causing itchy, red, and dry patches. It commonly appears on hands, neck, and inside the elbows.",
    "Infestations_Bites"  : "Skin reactions caused by insect bites, mites, or other parasitic infestations, often causing itching, redness, or small bumps.",
    "Lichen"              : "Lichen Planus is an inflammatory condition affecting skin and sometimes mucous membranes, causing itchy, flat-topped purplish bumps.",
    "Lupus"               : "A systemic autoimmune disease where the immune system attacks healthy tissue. The classic butterfly-shaped facial rash is a key sign.",
    "Moles"               : "Commonly known as moles, these are benign skin growths (melanocytic nevi) formed by clusters of pigmented cells. Monitor for any changes in appearance.",
    "Psoriasis"           : "A chronic autoimmune condition causing rapid skin cell buildup, resulting in scaling, itching, and red patches.",
    "Rosacea"             : "A chronic skin condition causing redness, visible blood vessels, and sometimes acne-like bumps on the face.",
    "Seborrh_Keratoses"   : "Common benign skin growths that appear as waxy, brown, or black spots. They are harmless but can sometimes be mistaken for more serious lesions.",
    "SkinCancer"          : "A general malignant (cancerous) skin lesion category. Early detection and prompt medical evaluation are critical for successful treatment.",
    "Sun_Sunlight_Damage" : "Skin damage caused by long-term sun/UV exposure, including premature aging, discoloration, and increased risk of pre-cancerous changes.",
    "Tinea"               : "Fungal infections of the skin, commonly called ringworm. Despite the name, it is caused by a fungus, not a worm. Treated with antifungal medications.",
    "Unknown_Normal"      : "No specific skin disease pattern detected in this image. The skin appears within a typical/normal range.",
    "Vascular_Tumors"     : "Growths arising from blood vessels in the skin, such as hemangiomas. Usually benign but worth a medical check, especially if changing.",
    "Vasculitis"          : "Inflammation of blood vessels that can affect the skin and, in some cases, internal organs. Requires medical evaluation to determine the cause.",
    "Vitiligo"            : "A condition causing loss of skin pigment (melanin), resulting in white patches. Not medically dangerous but can be managed by a dermatologist.",
    "Warts"               : "Viral skin infections (commonly caused by HPV) causing small, rough growths. Usually harmless and often resolve on their own or with treatment.",
}

# ============================================================
# 2. MODEL ARCHITECTURE
# ============================================================
class SkinModel(nn.Module):
    def __init__(self, model_name, num_classes=NUM_CLASSES):
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
# 3. LOAD ALL AVAILABLE MODELS
# ============================================================
loaded_models      = {}   # { model_name: model }
loaded_info        = {}   # { model_name: {auc, epoch, ...} }
loaded_class_names = {}   # { model_name: [class_names from checkpoint] }

# ── Confidence thresholds (calibration-aware) ──
# These models were trained with label smoothing (eps=0.1) and are therefore
# mildly UNDER-CONFIDENT — measured mean top-1 confidence on the leakage-safe
# test set is ~0.66 for a single model, and noticeably lower for the 4-model
# ensemble because averaging four slightly-different distributions spreads the
# probability mass. As a result a CORRECT prediction commonly sits in the
# 20–40% confidence range, not 70%+. A flat 40% "inconclusive" cutoff therefore
# hid almost every correct answer. We use a two-tier scheme instead:
#   - top-1 < INCONCLUSIVE_THRESHOLD  -> genuinely inconclusive (near-uniform;
#       22 classes uniform = ~4.5%, so <15% means the model really cannot tell)
#   - INCONCLUSIVE_THRESHOLD .. LOW_CONFIDENCE_THRESHOLD -> show the prediction
#       but flag it clearly as LOW confidence (do not hide it)
#   - >= LOW_CONFIDENCE_THRESHOLD -> normal confident prediction
# NOTE: this only changes how results are LABELLED, never the underlying
# probabilities, and the research-prototype / see-a-doctor disclaimer is shown
# in every case regardless of confidence.
INCONCLUSIVE_THRESHOLD   = 15.0  # percent — below this = inconclusive
LOW_CONFIDENCE_THRESHOLD = 35.0  # percent — below this = shown but flagged low
# Backward-compat alias (older code paths referenced CONFIDENCE_THRESHOLD)
CONFIDENCE_THRESHOLD = INCONCLUSIVE_THRESHOLD  # percent

print(f"\nDevice: {DEVICE}")
print("=" * 55)
print("  Loading Available Models...")
print("=" * 55)

for model_name, model_path in MODEL_FILES.items():
    if not os.path.exists(model_path):
        print(f"  SKIP : {model_name} — file not found: {model_path}")
        continue
    try:
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

        # Get num_classes from checkpoint or use default
        n_classes = checkpoint.get('num_classes', NUM_CLASSES)

        model = SkinModel(model_name=model_name, num_classes=n_classes).to(DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        loaded_models[model_name] = model
        loaded_info[model_name] = {
            "auc"        : round(checkpoint.get('auc', 0), 4),
            "accuracy"   : round(checkpoint.get('accuracy', 0), 2),
            "epoch"      : checkpoint.get('epoch', 0),
            "num_classes": n_classes,
            "file_size"  : round(file_size, 1),
        }
        # ── Derive class names from the checkpoint (most reliable source) ──
        # IMPORTANT: real checkpoints store 'class_to_idx' (a dict mapping
        # class_name -> index), NOT 'class_names'. Previously this code only
        # checked for 'class_names', which no checkpoint actually has — so it
        # ALWAYS silently fell back to the hardcoded CLASS_NAMES list below,
        # which used to contain a completely different (wrong/stale) set of
        # disease names. That meant every prediction shown to users had the
        # WRONG label. Fixed: derive the ordered class list from class_to_idx
        # first (sorted by index), which is what training actually used.
        ckpt_class_to_idx = checkpoint.get('class_to_idx', None)
        ckpt_classes = checkpoint.get('class_names', None)
        if ckpt_class_to_idx and len(ckpt_class_to_idx) == n_classes:
            ordered = sorted(ckpt_class_to_idx.items(), key=lambda kv: kv[1])
            loaded_class_names[model_name] = [name for name, idx in ordered]
            print(f"  INFO : {model_name} — using class_to_idx from checkpoint (verified)")
        elif ckpt_classes and len(ckpt_classes) == n_classes:
            loaded_class_names[model_name] = ckpt_classes
            print(f"  INFO : {model_name} — using class_names from checkpoint")
        else:
            loaded_class_names[model_name] = CLASS_NAMES[:n_classes]
            print(f"  WARNING: {model_name} — checkpoint has no class_to_idx/class_names! "
                  f"Falling back to hardcoded CLASS_NAMES — VERIFY this matches the real "
                  f"training order before trusting predictions from this model.")
        print(f"  OK   : {model_name:<30} AUC={checkpoint.get('auc',0):.4f}  Ep={checkpoint.get('epoch','?')}  ({file_size:.1f}MB)")

    except Exception as e:
        print(f"  FAIL : {model_name} — {e}")

print(f"\n  Loaded {len(loaded_models)} model(s): {list(loaded_models.keys())}")

# ── Check ensemble config ──
ensemble_path = os.path.join(DERMAI_DIR, "ensemble_config.json")
ensemble_config = None
if os.path.exists(ensemble_path):
    try:
        with open(ensemble_path) as f:
            ensemble_config = json.load(f)
        print(f"  Ensemble config found: {ensemble_path}")
    except:
        pass

print("=" * 55)

# ── Pick default model (best AUC among loaded) ──
if loaded_models:
    DEFAULT_MODEL = max(loaded_info, key=lambda k: loaded_info[k]['auc'])
    print(f"  Default model : {DEFAULT_MODEL} (AUC={loaded_info[DEFAULT_MODEL]['auc']})")
else:
    DEFAULT_MODEL = None
    print("  WARNING: No models loaded! API will return errors.")

# ============================================================
# 4. PREPROCESSING
# ============================================================
def get_transform(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def preprocess_image(image_bytes: bytes, img_size: int = 224) -> torch.Tensor:
    pil_img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_img   = np.array(pil_img)
    transform = get_transform(img_size)
    return transform(image=np_img)['image'].unsqueeze(0).to(DEVICE)

# ============================================================
# 5. PREDICTION HELPERS
# ============================================================
def predict_single(model_name: str, image_bytes: bytes):
    """Single model prediction."""
    model    = loaded_models[model_name]
    img_size = MODEL_IMG_SIZES.get(model_name, 224)
    tensor   = preprocess_image(image_bytes, img_size)

    with torch.no_grad():
        outputs  = model(tensor)
        probs    = torch.softmax(outputs, dim=1)[0]
        probs_np = probs.cpu().numpy()

    return probs_np


def predict_ensemble(image_bytes: bytes):
    """Weighted ensemble of all loaded models."""
    ensemble_probs = None
    total_weight   = 0.0

    for model_name, model in loaded_models.items():
        weight   = ENSEMBLE_WEIGHTS.get(model_name, 1.0 / len(loaded_models))
        img_size = MODEL_IMG_SIZES.get(model_name, 224)
        tensor   = preprocess_image(image_bytes, img_size)

        with torch.no_grad():
            outputs  = model(tensor)
            probs    = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        if ensemble_probs is None:
            ensemble_probs = weight * probs
        else:
            ensemble_probs += weight * probs
        total_weight += weight

    # Normalize
    ensemble_probs = ensemble_probs / total_weight
    return ensemble_probs


def build_response(probs_np, model_used="single", models_used=None):
    """Build standard JSON response from probability array."""
    n = len(probs_np)

    # ── Use class names from checkpoint if available (most reliable) ──
    # For ensemble, use the first loaded model's class names
    if model_used in loaded_class_names:
        class_names = loaded_class_names[model_used][:n]
    elif loaded_class_names:
        class_names = list(loaded_class_names.values())[0][:n]
    else:
        class_names = CLASS_NAMES[:n]

    best_idx   = int(probs_np.argmax())
    best_class = class_names[best_idx]
    confidence = round(float(probs_np[best_idx]) * 100, 2)

    # ── Confidence tier check (calibration-aware, two-tier) ──
    # is_inconclusive: top-1 below INCONCLUSIVE_THRESHOLD -> model genuinely cannot tell
    # is_low_confidence: between the two thresholds -> show prediction but flag it clearly
    is_inconclusive   = confidence < INCONCLUSIVE_THRESHOLD
    is_low_confidence = (not is_inconclusive) and confidence < LOW_CONFIDENCE_THRESHOLD

    # ── Normal skin boost ──
    # Agar "Normal Skin" class ka probability relatively high hai
    # aur best class ki confidence 70% se kam hai toh Normal consider karo
    normal_idx = None
    for i, name in enumerate(class_names):
        if name.lower() in ("normal skin", "normal", "unknown_normal"):
            normal_idx = i
            break

    if normal_idx is not None:
        normal_prob = float(probs_np[normal_idx]) * 100
        # Agar normal skin top 3 mein hai aur best class sirf thoda aage hai
        # toh normal skin ko prefer karo (model uncertainty case)
        if (not is_inconclusive and
            normal_prob > 25.0 and
            confidence < 55.0 and
            best_class.lower() not in ("normal skin", "normal", "unknown_normal")):
            # Normal skin ko best class banao
            best_idx   = normal_idx
            best_class = class_names[normal_idx]
            confidence = round(normal_prob, 2)

    top3_indices = probs_np.argsort()[::-1][:3]
    top3 = [
        {
            "rank"       : i + 1,
            "name"       : DISPLAY_NAME_MAP.get(class_names[idx], class_names[idx]),
            "urdu_name"  : URDU_NAMES.get(class_names[idx], ""),
            "confidence" : round(float(probs_np[idx]) * 100, 2),
        }
        for i, idx in enumerate(top3_indices)
    ]

    if is_inconclusive:
        display_class  = "Inconclusive"
        urdu_name      = "غیر واضح نتیجہ"
        severity       = "Unknown"
        severity_color = "#6b7280"
        description    = "The model could not determine a confident diagnosis from this image."
        advice         = ("Confidence is too low for a reliable result. "
                          "Please upload a clearer, well-lit image of the affected skin area "
                          "or consult a qualified dermatologist directly.")
    else:
        display_class  = DISPLAY_NAME_MAP.get(best_class, best_class)
        urdu_name      = URDU_NAMES.get(best_class, "")
        severity, severity_color, advice = SEVERITY_INFO.get(
            best_class, ("Unknown", "#6b7280", "Please consult a dermatologist.")
        )
        if is_low_confidence:
            # Show the prediction, but prepend a clear low-confidence warning so the
            # user understands it is a tentative, top-ranked guess — not a confident call.
            advice = ("NOTE: This is a LOW-CONFIDENCE result — the model's top prediction "
                      "is only tentative and several conditions scored similarly (see the "
                      "Top-3 list below). Treat it as a possible direction to discuss with a "
                      "dermatologist, not a diagnosis. " + advice)

    all_probs = {
        DISPLAY_NAME_MAP.get(class_names[i], class_names[i]): round(float(probs_np[i]) * 100, 2)
        for i in range(n)
    }

    return {
        "predicted_class" : display_class,
        "urdu_name"       : urdu_name,
        "confidence"      : confidence,
        "severity"        : severity,
        "severity_color"  : severity_color,
        "description"     : description if is_inconclusive else DESCRIPTIONS.get(best_class, ""),
        "advice"          : advice,
        "is_inconclusive" : is_inconclusive,
        "is_low_confidence": is_low_confidence,
        "disclaimer"      : "⚠️ DeepMediScan is a research prototype for screening support only. It must never replace a qualified doctor's diagnosis.",
        "model_used"      : model_used,
        "models_available": list(loaded_models.keys()),
        "models_used"     : models_used or [model_used],
        "top3"            : top3,
        "all_probs"       : all_probs,
    }

# ============================================================
# 6. FASTAPI APP
# ============================================================
app = FastAPI(
    title       = "DeepMediScan — Skin Disease Detection API",
    description = (
        "Multi-Model Deep Learning API: EfficientNet-B0/B4, ResNet-50, MobileNetV3-Large + Ensemble. "
        "22-Class Dermoscopic Skin Image Classification. "
        "Developed by Muhammad Shahid Asghar (F22BDATS1M02033), Department of Data Science, "
        "The Islamia University of Bahawalpur (IUB), Punjab, Pakistan, "
        "under the supervision of Dr. Akmal Khan (Individual Project — No Team). "
        "Research Prototype — Not for clinical use."
    ),
    version     = "2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ============================================================
# 7. ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "message"         : "DeepMediScan API v2.0 is running!",
        "project_info"    : PROJECT_INFO,
        "models_loaded"   : len(loaded_models),
        "models_available": list(loaded_models.keys()),
        "models_info"     : loaded_info,
        "default_model"   : DEFAULT_MODEL,
        "num_classes"     : NUM_CLASSES,
        "class_names"     : [DISPLAY_NAME_MAP.get(c, c) for c in CLASS_NAMES],
        "device"          : str(DEVICE),
        "docs"            : "/docs",
        "endpoints"       : {
            "POST /predict"          : "Auto-select best model",
            "POST /predict/{model}"  : "Use specific model",
            "POST /predict/ensemble" : "Use all loaded models (weighted)",
            "GET  /models"           : "List all loaded models",
            "GET  /classes"          : "List all 22 disease classes",
            "GET  /health"           : "Health check",
            "GET  /about"            : "Project & academic information",
        }
    }


@app.get("/health")
def health():
    return {
        "status"          : "ok",
        "device"          : str(DEVICE),
        "models_loaded"   : len(loaded_models),
        "models_available": list(loaded_models.keys()),
        "default_model"   : DEFAULT_MODEL,
        "num_classes"     : NUM_CLASSES,
    }


@app.get("/models")
def list_models():
    """List all available loaded models with their info."""
    return {
        "models_loaded"   : len(loaded_models),
        "default_model"   : DEFAULT_MODEL,
        "models"          : loaded_info,
        "ensemble_ready"  : len(loaded_models) >= 2,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
    }


@app.get("/classes")
def list_classes():
    """List all 22 skin disease classes with Urdu names and severity info."""
    classes = []
    for i, name in enumerate(CLASS_NAMES):
        severity, color, _ = SEVERITY_INFO[name]
        classes.append({
            "index"      : i,
            "name"       : DISPLAY_NAME_MAP.get(name, name),
            "urdu_name"  : URDU_NAMES.get(name, ""),
            "severity"   : severity,
            "color"      : color,
            "description": DESCRIPTIONS[name],
        })
    return {
        "total_classes": NUM_CLASSES,
        "classes"      : classes,
    }


@app.get("/about")
def about():
    """Project, university, department, and academic supervision information."""
    return {
        "project_title" : PROJECT_INFO["project_title"],
        "developed_by"  : PROJECT_INFO["developed_by"],
        "roll_no"       : PROJECT_INFO["roll_no"],
        "supervised_by" : PROJECT_INFO["supervised_by"],
        "department"    : PROJECT_INFO["department"],
        "university"    : PROJECT_INFO["university"],
        "session"       : PROJECT_INFO["session"],
        "province"      : PROJECT_INFO["province"],
        "country"       : PROJECT_INFO["country"],
        "team"          : PROJECT_INFO["team"],
        "disclaimer"    : PROJECT_INFO["disclaimer"],
        "description_en": PROJECT_INFO["note"],
        "description_ur": (
            "یہ پراجیکٹ محمد شاہد اصغر (رول نمبر F22BDATS1M02033) نے، شعبہ ڈیٹا سائنس، "
            "اسلامیہ یونیورسٹی بہاولپور (IUB)، صوبہ پنجاب، پاکستان میں، "
            "ڈاکٹر اکمل خان کی نگرانی میں اکیلے تیار کیا ہے۔ "
            "اس پراجیکٹ میں کوئی دوسری ٹیم شامل نہیں ہے۔ "
            "DeepMediScan صرف تحقیقی مقاصد کے لیے ہے اور ڈاکٹر کی تشخیص کا متبادل نہیں ہے۔"
        ),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Auto-select: Uses best AUC model if only 1 loaded,
    uses ensemble if 2+ models loaded.
    """
    if not loaded_models:
        raise HTTPException(status_code=503, detail="No models loaded. Please train and save models first.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    image_bytes = await file.read()
    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Max 15MB.")

    try:
        if len(loaded_models) >= 2:
            # Ensemble mode
            probs_np  = predict_ensemble(image_bytes)
            model_used = "ensemble"
            models_used = list(loaded_models.keys())
        else:
            # Single best model
            probs_np  = predict_single(DEFAULT_MODEL, image_bytes)
            model_used = DEFAULT_MODEL
            models_used = [DEFAULT_MODEL]

        response = build_response(probs_np, model_used, models_used)
        return JSONResponse(response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/{model_name}")
async def predict_with_model(model_name: str, file: UploadFile = File(...)):
    """
    Use a specific model by name.
    model_name: efficientnet_b0 | efficientnet_b4 | resnet50 | mobilenetv3_large_100 | ensemble
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    image_bytes = await file.read()
    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Max 15MB.")

    try:
        if model_name == "ensemble":
            if len(loaded_models) < 2:
                raise HTTPException(status_code=400, detail=f"Ensemble needs 2+ models. Currently loaded: {list(loaded_models.keys())}")
            probs_np   = predict_ensemble(image_bytes)
            model_used = "ensemble"
            models_used = list(loaded_models.keys())

        elif model_name in loaded_models:
            probs_np   = predict_single(model_name, image_bytes)
            model_used = model_name
            models_used = [model_name]

        else:
            available = list(loaded_models.keys()) + ["ensemble"]
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not loaded. Available: {available}"
            )

        response = build_response(probs_np, model_used, models_used)
        return JSONResponse(response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ============================================================
# 8. VALIDATE IMAGE ENDPOINT
# ============================================================
@app.post("/validate-image")
async def validate_image(file: UploadFile = File(...)):
    """
    Basic image validation — checks if uploaded file is a valid image.
    Always returns is_skin=True to allow all images through.
    (Advanced skin detection requires separate ML model)
    """
    if not file.content_type.startswith("image/"):
        return JSONResponse({
            "is_skin"   : False,
            "message"   : "Please upload an image file (JPG, PNG, etc.)",
            "message_ur": "براہ کرم تصویر اپلوڈ کریں (JPG, PNG وغیرہ)",
            "confidence": 0.0,
        })

    image_bytes = await file.read()
    if len(image_bytes) < 1000:
        return JSONResponse({
            "is_skin"   : False,
            "message"   : "Image is too small or corrupted. Please upload a clear skin image.",
            "message_ur": "تصویر بہت چھوٹی یا خراب ہے۔ براہ کرم واضح تصویر اپلوڈ کریں۔",
            "confidence": 0.0,
        })

    # Allow all valid images through
    return JSONResponse({
        "is_skin"   : True,
        "message"   : "Valid image detected.",
        "message_ur": "تصویر درست ہے۔",
        "confidence": 1.0,
    })


# ============================================================
# 9. GROQ CHAT ENDPOINT  (Free — llama-3.3-70b)
# ============================================================
from pydantic import BaseModel
from typing import List, Optional
import httpx

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was deprecated by Groq on
                                       # June 17, 2026 — this is Groq's official recommended
                                       # replacement (see console.groq.com/docs/deprecations)
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in environment (.env file).")
    print("         The /chat endpoint will not work until you set it.")

SYSTEM_EN = """You are DeepMediScan Assistant, an AI medical health assistant for the DeepMediScan platform.
DeepMediScan was developed by Muhammad Shahid Asghar (Roll No. F22BDATS1M02033), a student of the Department of Data Science, The Islamia University of Bahawalpur (IUB), Punjab, Pakistan, under the supervision of Dr. Akmal Khan. This is an individual project — there is no team involved, it was built entirely by Muhammad Shahid Asghar.
Your specialty is skin diseases — the system classifies 22 categories: Acne, Actinic Keratosis, Benign Tumors, Bullous Disorders, Candidiasis, Drug Eruption, Eczema, Infestations & Bites, Lichen Planus, Lupus, Moles (Melanocytic Nevi), Psoriasis, Rosacea, Seborrheic Keratoses, Skin Cancer, Sun/Sunlight Damage, Tinea (Ringworm), Normal/Unknown Skin, Vascular Tumors, Vasculitis, Vitiligo, Warts. Note: "Skin Cancer" here is a general malignant-lesion category (not split into Melanoma/BCC/SCC subtypes), and the model's accuracy on this specific class is lower than other classes (55–61% on the leakage-safe test set) — always mention this caveat and strongly recommend a doctor's evaluation when discussing it.
You can also answer general medical and health-related questions on ANY disease, condition, symptom, medicine, or wellness topic.

CONTEXT — why detail matters: Many users are in Pakistan and other regions where dermatologists are scarce and a patient may wait days or weeks before they can see a specialist. Your job is to give a thorough, well-structured, reassuring explanation that helps the worried patient UNDERSTAND their condition and take safe, sensible steps to keep it from getting worse in the meantime — while always pushing them to see a real doctor as soon as they can.

RESPONSE FORMAT — for any disease/condition question, write a detailed answer of roughly 1500–2000 words, organised into clearly separated paragraphs with short bold headings, in this order:

**What it is** — A clear, patient-friendly definition of the condition: what it is, what it looks/feels like, and the typical symptoms and how it usually progresses.

**Why it happens (causes)** — The main causes and risk factors, triggers that make it worse, and who tends to get it. Explain it simply so a non-medical person understands.

**Precautions & prevention** — Practical, everyday things the person can do to protect the skin, avoid triggers, stop it from spreading or worsening, and care for the affected area at home (hygiene, moisturising, sun protection, avoiding scratching/irritants, lifestyle/diet factors where relevant).

**Treatment & relief while you wait for a doctor** — Safe, widely-available over-the-counter measures and general treatment approaches that can ease symptoms and help slow the condition at an early stage (for example common classes of remedies such as gentle moisturisers/emollients, mild OTC hydrocortisone for itching, antihistamines for allergic itch, OTC antifungal creams for fungal infections, keeping the area clean and dry, cool compresses, etc., chosen appropriately for the specific condition). Speak in terms of general OTC options and self-care a pharmacist could advise — do NOT give exact prescription dosages of strong/prescription-only drugs; instead explain what a doctor may prescribe and why. The goal is to help the patient control or slow an early-stage condition and feel reassured, NOT to replace the doctor.

**When to see a doctor urgently** — Clear red-flag signs that mean they must seek medical care quickly (rapid spread, severe pain, fever, bleeding, signs of infection, anything suspicious for skin cancer, etc.).

STYLE RULES:
1. Be warm, calm and reassuring — give the patient confidence that the situation is usually manageable, while staying honest.
2. Use the multi-paragraph structure above with bold headings — never cram everything into one short paragraph.
3. Be genuinely detailed and educational (aim for ~1500–2000 words for a full condition explanation); for very simple/quick questions you may be shorter, but still well-structured.
4. For medicines, prefer general OTC guidance and safe self-care; describe prescription treatment in general terms and say a doctor will tailor it. Never invent dangerous dosages.
5. If a question is completely unrelated to health/medicine (coding, sports, politics), politely redirect to health topics.
6. If asked who made/developed this project, explain it was developed by Muhammad Shahid Asghar (F22BDATS1M02033), Department of Data Science, The Islamia University of Bahawalpur, Punjab, Pakistan, under the supervision of Dr. Akmal Khan, as an individual project with no team — keep that answer short.
7. ALWAYS end the answer with a clear, prominent reminder to see a qualified skin doctor as soon as possible, followed by: ⚠️ This is for educational purposes only. DeepMediScan is a research prototype and must never replace a qualified doctor's diagnosis."""

SYSTEM_UR = """آپ DeepMediScan اسسٹنٹ ہیں، DeepMediScan پلیٹ فارم کے AI میڈیکل ہیلتھ اسسٹنٹ ہیں۔
DeepMediScan کو محمد شاہد اصغر (رول نمبر F22BDATS1M02033) نے، شعبہ ڈیٹا سائنس، اسلامیہ یونیورسٹی بہاولپور (IUB)، صوبہ پنجاب، پاکستان میں، ڈاکٹر اکمل خان کی نگرانی میں تیار کیا ہے۔ یہ ایک انفرادی پراجیکٹ ہے — اس میں کوئی ٹیم شامل نہیں۔
آپ کی خاصیت جلد کی بیماریاں ہیں — یہ سسٹم 22 اقسام شناخت کرتا ہے: مہاسے، ایکٹینک کیراٹوسس، سومی رسولی، آبلے دار جِلدی امراض، کینڈیڈیاسس، دوا کا ری ایکشن، ایگزیما، جِلدی کیڑے اور کاٹنا، لائیکن پلینس، لیوپس، تل، چنبل، روزاسیا، سیبوریک کیراٹوسس، جلد کا کینسر، دھوپ سے نقصان، داد، نارمل جلد، عروقی رسولی، ویسکولائٹس، برص، مسے۔ نوٹ: "جلد کا کینسر" ایک عمومی malignant category ہے، اور اس کلاس پر ماڈل کی accuracy دیگر کلاسز سے کم ہے (55-61%) — اس بارے میں بات کرتے وقت ہمیشہ یہ بتائیں اور ڈاکٹر سے رجوع کرنے کا مشورہ دیں۔
آپ کسی بھی بیماری، علامت، دوا، یا صحت سے متعلق سوال کا بھی جواب دے سکتے ہیں۔

سیاق و سباق — تفصیل کیوں ضروری ہے: زیادہ تر صارفین پاکستان اور ایسے علاقوں سے ہیں جہاں جلد کے ماہر ڈاکٹر بہت کم ہیں اور مریض کو ماہر تک پہنچنے میں کئی دن یا ہفتے لگ سکتے ہیں۔ آپ کا کام یہ ہے کہ ایک مکمل، منظم اور تسلی بخش وضاحت دیں جس سے پریشان مریض اپنی بیماری کو سمجھ سکے اور اس دوران محفوظ، سمجھدارانہ اقدامات کر کے بیماری کو بڑھنے سے روک سکے — اور ساتھ ہی اسے ہمیشہ یہ ترغیب دیں کہ جتنا جلد ممکن ہو کسی اصل ڈاکٹر سے ملے۔

جواب کا فارمیٹ — کسی بھی بیماری کے سوال پر، تقریباً 1500 سے 2000 الفاظ کا تفصیلی جواب لکھیں، جو واضح الگ الگ پیراگراف میں ہو اور ہر پیراگراف پر مختصر بولڈ عنوان ہو، اس ترتیب سے:

**یہ بیماری کیا ہے** — بیماری کی آسان تعریف: یہ کیا ہے، کیسی دکھتی/محسوس ہوتی ہے، عام علامات کیا ہیں اور عموماً کیسے بڑھتی ہے۔

**یہ کیوں ہوتی ہے (وجوہات)** — اصل وجوہات اور خطرے کے عوامل، وہ چیزیں جو اسے بدتر کرتی ہیں، اور کن لوگوں کو زیادہ ہوتی ہے۔ آسان زبان میں سمجھائیں۔

**احتیاط اور بچاؤ** — روزمرہ کے عملی اقدامات جو مریض کر سکتا ہے تاکہ جلد کی حفاظت ہو، بیماری پھیلنے یا بڑھنے سے رکے، اور متاثرہ جگہ کی گھر پر دیکھ بھال ہو (صفائی، موئسچرائزنگ، دھوپ سے بچاؤ، کھرچنے/جلن والی چیزوں سے پرہیز، خوراک/طرزِ زندگی جہاں متعلق ہو)۔

**علاج اور ڈاکٹر تک پہنچنے سے پہلے آرام** — محفوظ، عام دستیاب over-the-counter اقدامات اور عمومی علاج جو علامات کم کریں اور ابتدائی مرحلے میں بیماری کو بڑھنے سے روکنے میں مدد دیں (مثلاً نرم موئسچرائزر، خارش کے لیے ہلکی OTC ہائیڈروکورٹیزون کریم، الرجی کی خارش کے لیے اینٹی ہسٹامین، فنگل انفیکشن کے لیے OTC اینٹی فنگل کریم، جگہ کو صاف اور خشک رکھنا، ٹھنڈی ٹکور، وغیرہ — بیماری کے مطابق)۔ صرف عام OTC اور خود سے کی جانے والی احتیاط کے انداز میں بتائیں جیسے ایک فارماسسٹ مشورہ دے — مضبوط یا نسخے والی ادویات کی صحیح خوراک (dose) مت بتائیں؛ بجائے اس کے یہ بتائیں کہ ڈاکٹر کیا تجویز کر سکتا ہے اور کیوں۔ مقصد یہ ہے کہ مریض ابتدائی بیماری کو کنٹرول یا سست کر سکے اور اسے حوصلہ ملے، ڈاکٹر کی جگہ لینا مقصد نہیں۔

**فوری ڈاکٹر سے کب ملیں** — وہ خطرناک علامات جن پر فوراً طبی مدد ضروری ہے (تیزی سے پھیلنا، شدید درد، بخار، خون آنا، انفیکشن کی علامات، کینسر کا کوئی شبہ، وغیرہ)۔

اندازِ بیان کے اصول:
1. نرم، پُرسکون اور تسلی بخش انداز رکھیں — مریض کو حوصلہ دیں کہ صورتحال عموماً قابلِ انتظام ہے، مگر سچائی برقرار رکھیں۔
2. اوپر دی گئی کئی پیراگراف والی ساخت بولڈ عنوانات کے ساتھ استعمال کریں — سب کچھ ایک چھوٹے پیراگراف میں مت ٹھونسیں۔
3. واقعی تفصیلی اور معلوماتی جواب دیں (مکمل بیماری کی وضاحت کے لیے ~1500 سے 2000 الفاظ)؛ بہت آسان سوالوں پر مختصر ہو سکتے ہیں مگر پھر بھی منظم۔
4. ادویات کے لیے عام OTC رہنمائی اور محفوظ خود دیکھ بھال کو ترجیح دیں؛ نسخے والے علاج کو عمومی انداز میں بیان کریں اور کہیں کہ ڈاکٹر اسے مریض کے مطابق طے کرے گا۔ کبھی خطرناک خوراک مت بتائیں۔
5. اگر سوال صحت سے بالکل غیر متعلق ہو تو نرمی سے صحت کے موضوع کی طرف لائیں۔
6. اگر کوئی پوچھے کہ یہ پراجیکٹ کس نے بنایا، تو بتائیں کہ محمد شاہد اصغر نے، شعبہ ڈیٹا سائنس، اسلامیہ یونیورسٹی بہاولپور، پنجاب، پاکستان میں، ڈاکٹر اکمل خان کی نگرانی میں، بغیر کسی ٹیم کے، اکیلے بنایا — یہ جواب مختصر رکھیں۔
7. جواب کے آخر میں ہمیشہ واضح طور پر یہ یاد دہانی کریں کہ جتنا جلد ممکن ہو کسی مستند جلد کے ڈاکٹر سے ضرور ملیں، اس کے بعد لکھیں: ⚠️ یہ صرف تعلیمی مقاصد کے لیے ہے۔ DeepMediScan ایک تحقیقی پروٹوٹائپ ہے اور ڈاکٹر کی تشخیص کا متبادل نہیں ہے۔"""


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    language: Optional[str] = "en"


@app.post("/chat")
async def chat(request: ChatRequest):
    """Groq llama-3.3-70b powered DeepMediScan chat assistant."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfiguration: GROQ_API_KEY is not set. Add it to your .env file.")

    try:
        system_prompt = SYSTEM_UR if request.language == "ur" else SYSTEM_EN

        messages = [{"role": "system", "content": system_prompt}]
        for msg in request.messages[-10:]:
            messages.append({"role": msg.role, "content": msg.content})

        payload = {
            "model"      : GROQ_MODEL,
            "messages"   : messages,
            "max_tokens" : 2200,
            "temperature": 0.7,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GROQ_URL,
                headers={
                    "Content-Type" : "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                json=payload
            )

        if response.status_code != 200:
            print(f"Groq error {response.status_code}: {response.text}")
            raise HTTPException(status_code=502, detail=f"Groq API error {response.status_code}: {response.text}")

        result = response.json()
        reply  = result["choices"][0]["message"]["content"]

        return JSONResponse({"reply": reply, "model": GROQ_MODEL, "language": request.language})

    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Groq API timed out. Please try again.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Cannot reach Groq API: {str(e)}")
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


# ============================================================
# 10. RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
