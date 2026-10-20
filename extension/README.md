# ScamShield AI — Manifest V3 Chrome Extension

This directory contains the source code for the **ScamShield AI Chrome Extension**, a Manifest V3 browser tool designed to detect employment scams, fake internship offers, and fraudulent recruiting tactics directly from job portals and career pages.

---

## 1. Key Capabilities

- **Single-Screen UI:** All primary features (Scan Current Page, Job Text Box, Check Pasted Text, and Screenshot Check) are accessible directly on the main screen without hidden dropdowns.
- **100% Offline Local Inference (Default Mode):**
  - **In-Browser Rule Engine:** Executes pattern matching for upfront fees, credential requests, WhatsApp/Telegram redirection, and unrealistic salary offers locally in JavaScript.
  - **In-Browser TF-IDF + Logistic Regression Model:** Implements sublinear TF-IDF scoring and logistic regression classification in vanilla JavaScript. Model weights and vocabulary are automatically synchronized by `train_model.py`.
  - Runs offline with zero network latency, zero external API calls, and zero CORS errors.
- **Optional Backend API Connection:**
  - Can connect to a local Flask backend (`http://127.0.0.1:5000`) or a deployed Google Cloud Run server for server-side Tesseract OCR screenshot extraction.
- **Smart Portal Disclaimer Stripping:** Automatically removes safety banners added by platforms like Naukri, Internshala, Indeed, LinkedIn, and Foundit, preventing false triggers.

---

## 2. Extension Files

```text
extension/
├── manifest.json       # Manifest V3 configuration (MV3 compliant)
├── popup.html          # Main popup UI layout
├── popup.css           # Styling, risk badges, and accessible typography
├── popup.js            # In-browser rule engine, ML inference, and API client
├── README.md           # This documentation
└── icons/              # Extension icons
    ├── icon16.png
    ├── icon48.png
    └── icon128.png
```

---

## 3. Installation Guide (Chrome, Brave, Edge)

1. Open Google Chrome and enter `chrome://extensions/` in the URL bar.
2. Enable **Developer mode** using the toggle switch in the top-right corner.
3. Click the **Load unpacked** button in the top-left toolbar.
4. Select the `extension/` folder in your local repository (`ScamShieldAI\extension`).
5. The **ScamShield AI** icon will appear in your browser extensions menu. Click the pin icon to keep it visible on your toolbar.

---

## 4. How to Use

### Method A: Scan Current Page
1. Navigate to any job post on LinkedIn, Indeed, Naukri, or Internshala.
2. Click the **ScamShield AI** icon in your toolbar.
3. Click **Scan Current Page**.
4. The extension extracts the job text from the page and places it into the review box so you can review what is being evaluated.
5. Click **Analyze Job Text** to see the assessment.

### Method B: Paste Job Description
1. Copy text from an email offer, messaging conversation, or job board.
2. Open the extension popup, paste into the text box, and click **Analyze Job Text**.

### Method C: Check Screenshot (OCR)
1. If you have an image or screenshot of a chat or letter, drag and drop it into the screenshot area.
2. If the local backend or Cloud Run server is running, Tesseract OCR will extract the text and run a full evaluation.

---

## 5. Permissions Rationale

The extension follows the principle of least privilege:

| Permission | Purpose |
| :--- | :--- |
| `activeTab` | Grants temporary access to the active browser tab only when the user explicitly clicks the extension or "Scan Current Page". |
| `scripting` | Executes a lightweight DOM script to extract text from job posting pages. |
| `host_permissions: ["<all_urls>"]` | Allows page scanning across any recruitment site, job board, or company career portal. |

---

## 6. Model Weight Synchronization

Whenever you retrain the machine learning model in Python:

```powershell
python train_model.py
```

The training script automatically extracts the vocabulary, IDF table, logistic regression weights, and intercept, updating `const ML_MODEL` inside `extension/popup.js`. No manual copy-pasting is required.
