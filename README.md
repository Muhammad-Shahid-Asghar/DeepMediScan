🚀 Live Demo: https://huggingface.co/spaces/Shahid-Asghar/DeepMediScan
<div align="center">

# 🔬 DeepMediScan

### AI-Powered Dermoscopic Skin Disease Classification

**A research prototype that classifies skin-lesion images into 22 categories using an ensemble of four deep-learning models, with a bilingual (English/Urdu) AI health assistant.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#-license)

</div>

> ⚠️ **Medical disclaimer:** DeepMediScan is a **research prototype for educational and screening-support purposes only**. It must **never** replace a qualified doctor's diagnosis. Always consult a licensed dermatologist.

---

## 📋 Overview

DeepMediScan takes a dermoscopic skin image and predicts the most likely condition out of **22 classes**, combining four CNN architectures into an equal-weight ensemble. It also includes a built-in **AI health assistant** (powered by the Groq API) that answers skin-disease questions in English and Urdu with detailed, structured guidance.

This project was developed as a **Final Year Project (FYP)** by **Muhammad Shahid Asghar** (Roll No. F22BDATS1M02033), Department of Data Science, **The Islamia University of Bahawalpur (IUB)**, Punjab, Pakistan, under the supervision of **Dr. Akmal Khan**.

---

## ✨ Features

- **22-class** skin disease classification
- **4-model ensemble** — EfficientNet-B0, EfficientNet-B4, ResNet-50, MobileNetV3-Large (equal-weight, 25 % each)
- **Calibration-aware confidence** — low-confidence predictions are clearly flagged rather than hidden
- **Bilingual AI assistant** (English + Urdu) with detailed, structured medical explanations
- **Image upload + camera capture + voice input**
- **Clean, responsive web UI** with light/dark themes and a settings panel
- **Per-analysis & chat history** kept on the device

---

## 🏆 Results (leakage-safe test set)

All models were retrained on a **leakage-safe re-split** after a near-duplicate audit (≈47 % dataset-wide near-duplication was found and removed).

| Model | Test Accuracy | Macro-F1 | AUC-ROC |
|---|---|---|---|
| EfficientNet-B0 | 78.23 % | 0.7209 | 0.9662 |
| EfficientNet-B4 | 75.97 % | 0.6894 | 0.9619 |
| ResNet-50 | 74.86 % | 0.6763 | 0.9623 |
| MobileNetV3-Large | 77.22 % | 0.7077 | 0.9685 |
| **Ensemble (all 4)** | **80.60 %** | **0.7473** | **0.9786** |

The ensemble adds a statistically significant **+2.36-point** accuracy gain over the best single model (McNemar test, *p* = 0.0004).

> Note: "Skin Cancer" is a general malignant-lesion category and is the weakest class (≈55–61 % accuracy) — the app discloses this and urges a doctor's evaluation.

---

## 🗂️ Project structure

```
DeepMediScan/
├── main.py                  # FastAPI backend (models, /predict, /chat, etc.)
├── index.html               # Web frontend (single-file UI)
├── requirements.txt         # Python dependencies
├── .env.example             # Template for your API key (copy to .env)
├── .gitignore
├── README.md
│
├── scripts/                 # (optional) reproducibility scripts
│   ├── full_metrics_eval.py     # per-model metrics + 95% CIs
│   ├── ensemble_eval.py         # ensemble + McNemar test
│   ├── generate_figures.py      # training/AUC/confusion figures
│   └── calibration_eval.py      # ECE / MCE / Brier
│
└── retrained_models/        # ⬇️ NOT in repo — download separately (see below)
    ├── best_efficientnet_b0_22class.pth
    ├── best_efficientnet_b4_22class.pth
    ├── best_resnet50_22class.pth
    └── best_mobilenetv3_large_100_22class.pth
```

---

## ⬇️ Model weights (download separately)

The trained `.pth` files are **too large for GitHub (200 MB+)** and are hosted externally:

> **Download link:** _add your link here_ (Hugging Face / Google Drive / GitHub Releases)

After downloading, place the four `.pth` files inside a `retrained_models/` folder in the project root (see structure above).

<details>
<summary>How to host the models (pick one)</summary>

- **Hugging Face (recommended for models):** create a free model repo at <https://huggingface.co/new>, upload the `.pth` files, and paste the link above.
- **GitHub Releases:** create a Release on this repo and attach the files (up to 2 GB each).
- **Google Drive:** upload, set "Anyone with the link", and paste the share link.
</details>

---

## 🚀 Getting started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/DeepMediScan.git
cd DeepMediScan
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> For a GPU build of PyTorch, install `torch`/`torchvision` from <https://pytorch.org/get-started/locally/> first.

### 4. Download the model weights
Place the four `.pth` files in `retrained_models/` (see the section above).

### 5. Configure your API key
```bash
# copy the template, then edit .env and paste your real key
cp .env.example .env
```
Get a free Groq API key at <https://console.groq.com/keys> and put it in `.env`:
```
GROQ_API_KEY=your_real_key_here
```

### 6. (Optional) Set a custom project path
By default the app looks for models and data in the folder where `main.py` lives, so you usually don't need to change anything. If your `retrained_models/` or dataset are in a different location, set a `DERMAI_DIR` variable in your `.env` file pointing to that folder.

### 7. Run the backend
```bash
python main.py
```
The API starts at **http://127.0.0.1:8000**.

### 8. Open the app
Open `index.html` in your browser (Chrome recommended for voice input). The settings panel lets you change the API address if needed.

---

## 🧪 Reproducing the results (optional)

The scripts in `scripts/` regenerate every reported number and figure from the trained checkpoints:

```bash
python scripts/full_metrics_eval.py     # per-model accuracy / F1 / AUC + 95% CIs
python scripts/ensemble_eval.py         # ensemble metrics + McNemar test
python scripts/calibration_eval.py      # ECE / MCE / Brier
python scripts/generate_figures.py      # training, per-class AUC & confusion figures
```
Edit the path/config block at the bottom of each script to point to your `retrained_models/` and dataset split before running.

---

## 🛠️ Tech stack

| Layer | Technology |
|---|---|
| Models | PyTorch, timm (EfficientNet, ResNet, MobileNetV3) |
| Backend | FastAPI, Uvicorn |
| Image pipeline | Albumentations, Pillow, NumPy |
| AI assistant | Groq API (OpenAI-compatible) |
| Frontend | HTML / CSS / JavaScript (single file) |
| Evaluation | scikit-learn, matplotlib |

---

## 📡 Main API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Health / welcome |
| `GET` | `/models` | List loaded models |
| `GET` | `/classes` | List the 22 classes |
| `POST` | `/predict` | Ensemble prediction on an uploaded image |
| `POST` | `/predict/{model_name}` | Single-model prediction |
| `POST` | `/chat` | AI health assistant (English/Urdu) |

---

## ⚠️ Disclaimer

DeepMediScan is **not a medical device** and is **not clinically validated**. It is an educational research prototype. Predictions — especially for the Skin Cancer class — can be wrong. **Always consult a qualified dermatologist** for any skin concern.

---

## 👤 Author

**Muhammad Shahid Asghar** — Roll No. F22BDATS1M02033
Department of Data Science, The Islamia University of Bahawalpur (IUB), Punjab, Pakistan
Supervised by **Dr. Akmal Khan**

*Individual Final Year Project — no team.*

---

## 📄 License

Released under the **MIT License** — see [`LICENSE`](LICENSE) for details.
