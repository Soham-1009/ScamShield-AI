# ScamShield AI — Explainable Fake Job & Internship Detector

ScamShield AI is a full-stack prototype and Manifest V3 Chrome Extension designed to help students and fresh graduates evaluate suspicious job advertisements and internship offers. It combines a **transparent, modular rule-based risk engine**, an **explainable deterministic template service**, **Tesseract OCR screenshot extraction**, and a **Scikit-learn TF-IDF + Logistic Regression classifier** (with an optional DistilBERT sequence classifier upgrade).

> **Important Epistemic & Safety Principle:** ScamShield AI never claims a job or internship offer is definitively fraudulent or guaranteed legitimate based solely on a machine learning score or pattern match. Rule-based warning evidence and statistical ML predictions are displayed separately, and every assessment provides actionable verification steps. No generative AI or LLM APIs (such as Gemini, OpenAI, or Ollama) are used anywhere in the system.

---

## 1. Architecture & Key Features

- **Transparent Rule-Based Risk Engine (`scamshield/risk_rules.py`)**
  - Detects warning patterns across six core recruitment risk categories:
    1. Upfront registration, training, equipment, or security fees
    2. Requests for bank login credentials, payment card details, or premature identity documents
    3. Unrealistic salary promises and guaranteed placement without interviews
    4. Excessive urgency or countdown pressure to accept/pay immediately
    5. Missing or unverifiable employer information (including personal webmail recruiter addresses)
    6. Requests to communicate or interview exclusively via unofficial channels (WhatsApp, Telegram, Signal)
  - Extracts the **exact verbatim phrases** that triggered each rule and pairs them with a clear explanation and a concrete verification step.
  - Never labels an offer `"High Risk"` without explicit rule-based evidence, and never labels an offer `"Legitimate"` simply because zero configured rules triggered.

- **Deterministic Explanation Service (`scamshield/explanation_service.py`)**
  - Generates structured JSON explanations using predefined templates and matched evidence—100% free of LLM dependencies or hallucinated company verifications.
  - Strictly separates **explicit matched evidence** from **general contextual indicators**.

- **Machine Learning Classification Pipeline (`train_model.py` & `scamshield/ml_detector.py`)**
  - Trains a CPU-friendly `Pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(class_weight="balanced"))` on the public Fake Job Postings dataset.
  - Uses a stratified train/test split and fits the TF-IDF vectorizer strictly on the training split to prevent data leakage.
  - Evaluates fraud-class precision, recall, F1-score, confusion matrix, and average precision (PR-AUC), saving the model to `models/fake_job_model.joblib` and metadata to `models/model_metadata.json`.
  - If no trained model artifact is present, the backend and extension explicitly report that ML predictions are unavailable rather than displaying a fabricated `0/100` score.

- **Optional DistilBERT Sequence Classifier (`train_distilbert.py`)**
  - Fine-tunes `distilbert-base-uncased` with class-weighted cross-entropy loss using a train/validation/test split and compares its held-out test metrics directly against the TF-IDF + Logistic Regression baseline on the same test set.

- **Tesseract OCR Screenshot Analysis (`scamshield/ocr_service.py`)**
  - Extracts text from uploaded PNG, JPG, WEBP, BMP, or TIFF offer screenshots (`POST /api/analyze-image`) and passes the extracted text through the full analysis pipeline.

- **Manifest V3 Chrome Extension (`extension/`)**
  - Uses `activeTab` and `chrome.scripting.executeScript` only when the user clicks **Scan Current Page**, placing extracted text into the review box so the user controls what is submitted.
  - Displays Rule-Based Assessment, Statistical ML Classifier, Matched Warning Evidence, Deterministic Explanation, and Recommended Verification Steps in distinct sections.

---

## 2. Project Structure

```text
ScamShieldAI/
├── app.py
├── train_model.py
├── train_distilbert.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── scamshield/
│   ├── __init__.py
│   ├── _wsgi_fallback.py
│   ├── ml_detector.py
│   ├── risk_rules.py
│   ├── explanation_service.py
│   ├── ocr_service.py
│   └── validation.py
├── extension/
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.css
│   ├── popup.js
│   └── icons/
│       ├── icon16.png
│       ├── icon48.png
│       └── icon128.png
├── tests/
│   ├── __init__.py
│   ├── test_rules.py
│   ├── test_api.py
│   └── test_validation.py
└── deployment/
    └── Dockerfile
```

---

## 3. Windows PowerShell Setup & Execution Guide

Run the following exact commands in **Windows PowerShell** from the project root directory (`ScamShieldAI`).

### Step 1: Create a Virtual Environment & Install Python Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Step 2: Install Tesseract OCR (Required for Screenshot Analysis)

```powershell
# Install Tesseract OCR via Windows Package Manager (winget)
winget install --id UB-Mannheim.TesseractOCR

# Verify Tesseract is on your PATH (or set TESSERACT_CMD in .env)
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
& $env:TESSERACT_CMD --version
```

### Step 3: Configure the Dataset & Train the Baseline ML Model

Download the public Fake Job Postings CSV dataset (containing columns `title`, `company_profile`, `description`, `requirements`, `benefits`, and binary target `fraudulent` in `{0, 1}`) and place it at `data\fake_job_postings.csv`:

```powershell
# Place your dataset at data\fake_job_postings.csv, then run:
python train_model.py --data-path data\fake_job_postings.csv --model-path models\fake_job_model.joblib --metadata-path models\model_metadata.json
```

*Note:* If `data\fake_job_postings.csv` is not present, `train_model.py` prints setup instructions and exits cleanly without generating a fake model. The Flask backend and Chrome extension will still run in rule-based + OCR mode and clearly state that the ML model is not loaded.

### Step 4 (Optional): Train & Compare the DistilBERT Sequence Classifier

```powershell
pip install torch transformers
python train_distilbert.py --data-path data\fake_job_postings.csv --output-dir models\distilbert_fake_job --report-path models\distilbert_comparison.json --epochs 2
```

### Step 5: Run the Automated Test Suite

```powershell
# Run with pytest
pytest -v

# Or run with Python's built-in unittest runner
python -m unittest discover -s tests -v
```

### Step 6: Start the Flask Backend Server

```powershell
$env:SCAMSHIELD_HOST = "127.0.0.1"
$env:SCAMSHIELD_PORT = "5000"
python app.py
```

Verify the health endpoint in a second PowerShell terminal:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/health" -Method Get | ConvertTo-Json -Depth 5
```

### Step 7: Load the Chrome Extension in Google Chrome

1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** using the toggle in the top-right corner.
3. Click **Load unpacked** in the top-left action bar.
4. Select the `extension` folder inside this project (`ScamShieldAI\extension`).
5. Pin **ScamShield AI** to your Chrome toolbar, open any job listing page, click the extension icon, and use **Scan Current Page**, paste text, or upload a screenshot.

---

## 4. API Reference

### `GET /api/health`
Returns service status, rule engine rule count, ML model availability, and Tesseract OCR status. Contains no LLM configuration.

### `POST /api/analyze`
Request body (`application/json`):
```json
{
  "job_text": "Data Entry Intern. Pay a refundable registration fee of $100 via wire transfer. Contact HR on WhatsApp only.",
  "source_url": "https://example.com/jobs/123"
}
```
Returns `200 OK` with separated `rule_based_assessment`, `ml_prediction`, `experimental_triage`, `triggered_rules`, `exact_matched_phrases`, `explanation`, `safety_recommendations`, and `processing_status`.

### `POST /api/analyze-image`
Request body (`multipart/form-data`):
- `image`: Screenshot file (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`, max 5 MB).
- `source_url` (optional): Source page URL string.

Returns `200 OK` with `ocr_extraction` metadata alongside the full analysis structure, `400`/`422` for unreadable or blank images, or `503` if Tesseract OCR is not installed.

---

## 5. Optional Google Cloud Run Deployment

To deploy the backend container to Google Cloud Run:

1. **Train the model locally first** (if you want ML predictions enabled in Cloud Run) so that `models/fake_job_model.joblib` and `models/model_metadata.json` exist before building the image, or deploy in rule-based + OCR mode.
2. **Build and deploy using Google Cloud SDK**:

```powershell
# Authenticate and set your GCP project
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID

# Build container image using Cloud Build and deployment/Dockerfile
gcloud builds submit --tag gcr.io/YOUR_GCP_PROJECT_ID/scamshield-ai:1.0.0 -f deployment/Dockerfile .

# Deploy to Cloud Run
gcloud run deploy scamshield-ai `
  --image gcr.io/YOUR_GCP_PROJECT_ID/scamshield-ai:1.0.0 `
  --platform managed `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 1Gi `
  --cpu 1 `
  --set-env-vars "SCAMSHIELD_RATE_LIMIT_PER_MINUTE=60,SCAMSHIELD_MAX_TEXT_LENGTH=25000"
```

3. **Connect the Chrome Extension**:
   - Open the ScamShield AI extension popup, expand **Configure Backend Server URL**, enter your Cloud Run HTTPS URL, and click **Apply**.
   - Note that when using a remote Cloud Run URL, submitted job text and screenshots are transmitted over HTTPS to that service for in-memory processing (and are never stored or logged).

---

## 6. Known Limitations & Security Notes

- **Estimates, Not Legal Proof**: Neither rule-based triggers nor statistical ML scores prove that a job listing is fraudulent or legitimate. Sophisticated scams may avoid common trigger phrases, and legitimate listings may occasionally use urgent phrasing.
- **OCR Accuracy**: Screenshot extraction via Tesseract depends on image resolution, contrast, and layout. Multi-column screenshots or stylized fonts may introduce character recognition errors.
- **No Persistent Storage**: Submitted job descriptions and uploaded screenshots are processed strictly in memory and are never written to disk or application logs.
