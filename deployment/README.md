# ScamShield AI — Deployment Guide

This directory contains the container configuration for deploying the ScamShield AI backend service to Google Cloud Run or any Docker-compatible container hosting environment.

---

## 1. Container Architecture

The Docker container is built on `python:3.11-slim` and includes:
- **Tesseract OCR Engine:** Installed via `apt-get` for server-side screenshot text extraction (`POST /api/analyze-image`).
- **Gunicorn WSGI Server:** Multi-threaded production server configured with 2 workers and 4 threads.
- **Non-Root Execution:** Runs under a dedicated `appuser` service account for enhanced container security.
- **Preloaded ML Models:** Reads trained model artifacts (`models/fake_job_model.joblib` and `models/model_metadata.json`) if pre-trained prior to image build.

---

## 2. Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `8080` | Port listened to by Gunicorn (Cloud Run standard). |
| `SCAMSHIELD_HOST` | `0.0.0.0` | Host bind address for incoming connections. |
| `SCAMSHIELD_PORT` | `5000` | Port used when running via `python app.py`. |
| `SCAMSHIELD_RATE_LIMIT_PER_MINUTE` | `60` | Sliding window rate limit per client IP address. |
| `SCAMSHIELD_CORS_ORIGIN` | unset | Exact production Chrome extension origin to allow. CORS is not authentication. |
| `SCAMSHIELD_MAX_TEXT_LENGTH` | `25000` | Maximum character length for job description analysis. |
| `SCAMSHIELD_OCR_TIMEOUT_SECONDS` | `30` | Maximum time allowed for one screenshot OCR operation. |
| `SCAMSHIELD_MODEL_PATH` | `models/fake_job_model.joblib` | Path to fitted Scikit-learn Joblib pipeline. |
| `SCAMSHIELD_MODEL_METADATA_PATH` | `models/model_metadata.json` | Path to model metadata and diagnostics JSON. |
| `TESSERACT_CMD` | System PATH | Explicit binary path for Tesseract OCR executable. |

---

## 3. Local Docker Build & Run

### Step 1: Pre-train the ML Model (Optional)
If you want statistical ML predictions enabled in your container, train from reviewed data with an employer/source/campaign group field first. The repository's 120-row local dataset has no such field; only use it for an explicitly exploratory random split:

```powershell
python train_model.py --data-path data\reviewed_jobs.csv --group-column employer_group
```

### Step 2: Build the Container Image
From the repository root:

```powershell
docker build -t scamshield-ai:latest -f deployment/Dockerfile .
```

### Step 3: Run the Container

```powershell
docker run -p 8080:8080 -e SCAMSHIELD_RATE_LIMIT_PER_MINUTE=100 scamshield-ai:latest
```

Test the health check:
```powershell
curl http://localhost:8080/api/health
```

---

## 4. Google Cloud Run Deployment

Deploy directly from source using the Google Cloud SDK:

```powershell
# Authenticate and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build container image with Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/scamshield-ai:1.1.0 -f deployment/Dockerfile .

# Deploy to Cloud Run
gcloud run deploy scamshield-ai `
  --image gcr.io/YOUR_PROJECT_ID/scamshield-ai:1.1.0 `
  --platform managed `
  --region us-central1 `
  --no-allow-unauthenticated `
  --memory 1Gi `
  --cpu 1 `
  --set-env-vars "SCAMSHIELD_CORS_ORIGIN=chrome-extension://YOUR_EXTENSION_ID,SCAMSHIELD_RATE_LIMIT_PER_MINUTE=60,SCAMSHIELD_MAX_TEXT_LENGTH=25000"
```

This deploys Cloud Run as a private service. Do not switch it to unauthenticated access for convenience: the extension cannot safely contain a shared server secret. If users need remote screenshot OCR, put an authenticated API gateway in front of the service and apply edge rate limits (for example, Cloud Armor) there. The Flask limiter is in-memory per process and is only a secondary local safeguard; it is not a service-wide public abuse control. Configure the exact production extension ID in `SCAMSHIELD_CORS_ORIGIN`; CORS still does not replace gateway authentication.
