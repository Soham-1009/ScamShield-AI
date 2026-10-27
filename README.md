# ScamShield AI — Explainable Fake Job & Internship Detector

ScamShield AI is an open-source, full-stack prototype and Manifest V3 Chrome Extension designed to protect students, job seekers, and fresh graduates from employment scams, predatory training fees, fake check fraud, and identity theft.

It combines a **transparent, modular rule-based risk engine**, an **explainable deterministic template explanation service**, **Tesseract OCR screenshot extraction**, and a **regularized Scikit-learn TF-IDF + Logistic Regression classifier** (with an optional DistilBERT sequence classifier upgrade).

> [!IMPORTANT]
> **Core Epistemic & Safety Principle:** ScamShield AI never claims a job or internship offer is definitively fraudulent or guaranteed legitimate based solely on a machine learning score or pattern match. Rule-based warning evidence and statistical ML predictions are displayed separately, and every assessment provides actionable verification steps. **Zero generative AI or LLM APIs (such as Gemini, OpenAI, or Ollama) are used anywhere in the system.**

---

## 1. Architecture & Key Features

### Transparent Rule-Based Risk Engine (`scamshield/risk_rules.py`)
- Analyzes job postings across six core recruitment threat vectors:
  1. **Upfront Fees:** Mandatory registration, training, certification, background check, or equipment deposits (supports `$`, `£`, `€`, `₹`/`Rs.`, UPI, GPay, PhonePe, Paytm, Zelle, Venmo, CashApp).
  2. **Credential & Identity Theft:** Requests for bank account logins, payment card CVV/PIN, ATM credentials, or scans of Aadhaar, PAN, SSN, or passport before an interview.
  3. **Unrealistic Promises:** Inflated daily/weekly earnings for simple tasks (e.g. `$800/day` for copy-paste typing) and guaranteed job placements without interviews.
  4. **Urgency & Rush Pressure:** Countdown tactics (`"only 2 slots left"`, `"confirm within 24 hours or lose offer"`).
  5. **Hidden / Personal Contact Details:** Anonymous employers (`"leading MNC"`, `"undisclosed client"`) and free webmail addresses (`@gmail.com`, `@yahoo.com`).
  6. **Unofficial Communication Channels:** Insisting on interviewing or communicating exclusively via WhatsApp or Telegram.
- **Smart Portal Disclaimer Stripping:** Automatically removes safety disclaimers inserted by recruitment portals (Naukri, Internshala, Indeed, LinkedIn, Foundit) so portal warnings are not mistaken for scam text.
- **Verbatim Evidence Extraction:** Extracts the exact matching phrases from the listing, paired with human-readable rationale and actionable next steps.

### Deterministic Explanation Service (`scamshield/explanation_service.py`)
- Builds structured explanations using predefined, deterministic templates.
- Guarantees zero hallucinations, consistent wording, and immediate latency.
- Clearly separates **explicit matched evidence** from **contextual indicators**.

### Regularized Machine Learning Classifier (`train_model.py` & `scamshield/ml_detector.py`)
- **Pipeline:** `Pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(C=1.0, class_weight="balanced"))`.
- **Anti-Overfitting & Data Leakage Prevention:**
  - Sanitizes template noise tokens (e.g. `(Batch X)` tags) and deduplicates text records prior to splitting.
  - Stratified 80/20 train/test split applied **before** TF-IDF vectorizer fitting.
  - 5-Fold Stratified Cross-Validation on the training split evaluates both F1-score and cross-validated log-loss to ensure well-calibrated, non-extreme probabilities.
  - Automatically calculates and logs the **generalization gap** (`train_f1 - test_f1`) in `models/model_metadata.json`.
- **Offline Extension Weight Synchronization:**
  - `train_model.py` automatically extracts vocabulary, IDF values, regression coefficients, and intercept directly into `extension/popup.js`.
  - When model files are missing, the system explicitly reports `available: false` rather than fabricating predictions.

### Tesseract OCR Screenshot Analysis (`scamshield/ocr_service.py`)
- Extracts text from uploaded screenshot images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`, max 5 MB).
- Automatic discovery of Tesseract executables across standard Windows paths (`C:\Program Files\Tesseract-OCR\tesseract.exe`, `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`, `LOCALAPPDATA\Programs\Tesseract-OCR\tesseract.exe`) or via custom `TESSERACT_CMD`.
- Decompression bomb safety guard (`MAX_IMAGE_PIXELS = 25_000_000`).
- Image preprocessing (grayscale, Otsu thresholding, contrast enhancement) for reliable extraction from chat app screenshots and email offer letters.

### Manifest V3 Chrome Extension (`extension/`)
- **Single-Screen UI:** All primary actions (Scan Current Page, Job Text Review Box, Check Pasted Text, and Screenshot Check) are directly visible on the main screen without hidden dropdowns.
- **Dual-Mode Inference:**
  - **In-Browser Local Engine (Default):** Runs the complete rule-based risk checks and the regularized TF-IDF + Logistic Regression model 100% locally in Chrome. Works offline with zero network latency and zero CORS configuration.
  - **Backend API Mode (Optional):** Can connect to the local Flask backend (`http://127.0.0.1:5000`) or a deployed Google Cloud Run instance for full OCR and server-side analysis.
- **Security & Privacy:** Uses `activeTab` and `chrome.scripting.executeScript` only on user click. Text is loaded into the editable box for review before analysis.

---

## 2. Project Structure

```text
ScamShieldAI/
├── app.py                      # Flask REST API server & rate limiting
├── train_model.py              # ML training, CV hyperparameter tuning, & extension sync
├── train_distilbert.py         # Optional DistilBERT transformer training & evaluation
├── requirements.txt            # Python dependencies (Flask, scikit-learn, pytesseract, etc.)
├── .env.example                # Environment variable configuration template
├── .gitignore                  # Git ignore rules (virtual environments, caches, datasets)
├── README.md                   # Complete system documentation
├── data/                       # Dataset directory (contains data/.gitkeep)
│   └── fake_job_postings.csv   # Training dataset (120 unique balanced listings)
├── models/                     # Model artifacts (contains models/.gitkeep)
│   ├── fake_job_model.joblib   # Fitted Scikit-learn TF-IDF + LogReg pipeline
│   └── model_metadata.json     # Training metrics, CV scores, & generalization diagnostics
├── scamshield/                 # Core business logic package
│   ├── __init__.py             # Package exports and version metadata
│   ├── ml_detector.py          # ML model loader & inference wrapper
│   ├── risk_rules.py           # 6-category regex risk rules & weight calculations
│   ├── explanation_service.py  # Deterministic templated explanation generator
│   ├── ocr_service.py          # Tesseract OCR extraction & preprocessing
│   └── validation.py           # Request validation, URL sanitization, & image guards
├── extension/                  # Manifest V3 Chrome Extension
│   ├── manifest.json           # Extension manifest (MV3, activeTab, scripting, host_permissions)
│   ├── popup.html              # Main extension UI layout
│   ├── popup.css               # Clean styling & risk level indicators
│   ├── popup.js                # In-browser rule engine, ML inference, & backend client
│   └── icons/                  # Extension icons (16px, 48px, 128px)
├── tests/                      # Automated test suite (31 tests)
│   ├── __init__.py
│   ├── test_rules.py           # Rule engine unit tests & disclaimer preservation
│   ├── test_api.py             # Flask API endpoints, OCR, & rate limit tests
│   └── test_validation.py      # Input validation & image boundary tests
└── deployment/                 # Deployment configurations
    └── Dockerfile              # Container definition for Google Cloud Run
```

---

## 3. Windows PowerShell Setup & Execution Guide

Follow these step-by-step instructions in **Windows PowerShell** from the project root (`c:\Users\soham\Desktop\Soham\ScamShieldAI`).

### Step 1: Create Virtual Environment & Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Step 2: Install Tesseract OCR (For Screenshot Analysis)

```powershell
# Install via Windows Package Manager (winget)
winget install --id UB-Mannheim.TesseractOCR

# ScamShield AI automatically discovers Tesseract at:
# C:\Program Files\Tesseract-OCR\tesseract.exe
# If installed in a non-standard location, set TESSERACT_CMD in your .env:
# TESSERACT_CMD=C:\custom\path\to\tesseract.exe
```

### Step 3: Train the Machine Learning Classifier

To train or retrain the baseline model and synchronize the weights with the Chrome extension:

```powershell
python train_model.py --data-path data\fake_job_postings.csv --model-path models\fake_job_model.joblib --metadata-path models\model_metadata.json
```

Output highlights:
- Evaluates 5-fold Stratified Cross-Validation on the training split.
- Automatically selects the optimal regularized $C$ parameter.
- Evaluates held-out test precision, recall, F1, and PR-AUC.
- Verifies generalization gap (`train_f1 - test_f1 = 0.0000`).
- Synchronizes model weights directly to `extension/popup.js`.

### Step 4: Run the Automated Test Suite

```powershell
# Run with pytest (all 31 unit, OCR, and API tests)
pytest -v

# Or run with Python unittest
python -m unittest discover -s tests -v
```

### Step 5: Start the Flask Backend Server

```powershell
$env:SCAMSHIELD_HOST = "127.0.0.1"
$env:SCAMSHIELD_PORT = "5000"
python app.py
```

Test the health check in a separate terminal:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/health" -Method Get | ConvertTo-Json -Depth 5
```

### Step 6: Load the Chrome Extension

1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Turn on **Developer mode** (toggle in the top-right corner).
3. Click **Load unpacked** (top-left action bar).
4. Select the `extension/` directory inside this project repository.
5. Pin **ScamShield AI** to your Chrome toolbar.
6. Open any job posting on LinkedIn, Indeed, Naukri, or Internshala, click the extension icon, and click **Scan Current Page** or paste text directly.

---

## 4. API Reference

### `GET /api/health`
Returns service status, active rule count, ML model version and availability, and Tesseract OCR installation status.

**Sample Response:**
```json
{
  "status": "healthy",
  "service": "ScamShield AI",
  "rule_engine": {
    "available": true,
    "rule_count": 8
  },
  "machine_learning": {
    "available": true,
    "model_type": "TF-IDF + Logistic Regression (scikit-learn)",
    "model_version": "1.1.0-tfidf-regularized-logreg"
  },
  "ocr_service": {
    "available": true,
    "engine": "Tesseract OCR"
  }
}
```

### `POST /api/analyze`
Analyzes raw text from a job advertisement or email offer.

**Request (`application/json`):**
```json
{
  "job_text": "Work from home data entry intern. Pay a refundable registration fee of $120 via Telegram/UPI. Direct offer letter without interview.",
  "source_url": "https://example.com/jobs/42"
}
```

**Response (`200 OK`):**
```json
{
  "rule_based_assessment": {
    "risk_level": "High Risk",
    "risk_score": 90,
    "triggered_rule_count": 3
  },
  "ml_prediction": {
    "available": true,
    "predicted_label": 1,
    "predicted_class": "fraudulent_pattern_detected",
    "fraud_probability_score": 0.6855,
    "fraud_score_pct": 68.6,
    "model_version": "1.1.0-tfidf-regularized-logreg",
    "disclaimer": "Statistical ML predictions measure text similarity against historical training data and are NOT proof that a specific job offer is fraudulent or legitimate."
  },
  "triggered_rules": [
    {
      "id": "RULE_UPFRONT_FEE",
      "severity": "high",
      "weight": 55,
      "exact_matched_phrases": ["refundable registration fee of $120"],
      "reason": "Asks you to pay money (registration, training, or security fee). Real companies pay you — they do not charge you to join.",
      "action": "Never pay any fee or deposit for a job or internship, even if they call it refundable."
    }
  ],
  "explanation": {
    "summary": "This job listing exhibits 3 high-risk warning indicator(s)...",
    "matched_evidence": ["refundable registration fee of $120"],
    "actionable_steps": ["Never send money, registration deposits, or gift cards to secure a job."]
  }
}
```

### `POST /api/analyze-image`
Uploads a screenshot (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`, max 5 MB) for OCR text extraction and full risk analysis.

**Request (`multipart/form-data`):**
- `image`: Image file binary
- `source_url` (optional): Source URL string

**Response (`200 OK`):**
Returns the full analysis response alongside `ocr_extraction` details (`extracted_character_count`, `ocr_engine`, `line_count`).

---

## 5. Google Cloud Run Deployment

To deploy the backend to Google Cloud Run:

```powershell
# 1. Authenticate with Google Cloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Build the container image via Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/scamshield-ai:1.1.0 -f deployment/Dockerfile .

# 3. Deploy to Cloud Run
gcloud run deploy scamshield-ai `
  --image gcr.io/YOUR_PROJECT_ID/scamshield-ai:1.1.0 `
  --platform managed `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 1Gi `
  --cpu 1 `
  --set-env-vars "SCAMSHIELD_RATE_LIMIT_PER_MINUTE=60,SCAMSHIELD_MAX_TEXT_LENGTH=25000"
```

To connect the extension to your Cloud Run backend, click **Configure Backend Server URL** in the extension popup and input your Cloud Run HTTPS URL.

---

## 6. Security Guarantees & Privacy Principles

- **Zero Generative AI:** Does not send user text or screenshots to LLM endpoints (OpenAI, Gemini, Ollama, etc.). Eliminates hallucinated facts, data leaks, and token costs.
- **In-Memory Processing:** Submitted job descriptions and uploaded screenshots are analyzed strictly in volatile memory. No text or images are saved to disk, databases, or logs.
- **Explainable Assessments:** Every risk score is accompanied by the verbatim phrase that triggered it, so users can verify the evidence themselves.
- **Empowerment Over Judgment:** ScamShield AI acts as an educational and triage tool, providing actionable verification steps rather than issuing unverifiable verdicts.
