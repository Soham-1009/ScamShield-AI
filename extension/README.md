# ScamShield AI — Manifest V3 Chrome Extension

This directory contains the source code for the **ScamShield AI Chrome Extension**, a Manifest V3 browser tool designed to detect employment scams, fake internship offers, and fraudulent recruiting tactics directly from job portals and career pages.

---

## 1. Key Capabilities

- **Single-Screen UI:** All primary features (Scan Current Page, Job Text Box, Check Pasted Text, and Screenshot Check) are accessible directly on the main screen without hidden dropdowns.
- **Local Inference by Default:**
  - **In-Browser Rule Engine:** Executes pattern matching for upfront fees, credential requests, WhatsApp/Telegram redirection, and unrealistic salary offers locally in JavaScript.
  - **In-Browser TF-IDF + Logistic Regression Model:** Implements sublinear TF-IDF scoring and logistic regression classification in vanilla JavaScript. Model weights and vocabulary are automatically synchronized by `train_model.py`.
  - Page and pasted-text checks run locally after the user starts a scan. They do not send listing text to the backend.
- **Optional Backend API Connection:**
  - Screenshot OCR sends the selected image to the configured backend only after a confirmation prompt. Remote backends must use HTTPS and require a per-origin browser permission.
  - The model score is an experimental classifier signal, not a calibrated probability or a fraud verdict.
- Rule scores count only configured rule matches. A zero score means no configured rule matched; it does not mean the listing is safe.
- Page scans show which key fields were captured. Missing title, employer, or description data is shown as **Not Enough Information** when no rule evidence was found. Screenshot OCR always needs a manual comparison with the source image.
- Rule findings show matched wording and surrounding context. Compensation cautions such as an unpaid role prompt verification and do not independently indicate fraud.
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
3. Click **Scan & Check This Job Page**.
4. Review the extracted text and assessment. The extension may capture only the description on some sites; confirm the title, employer, and terms are included before relying on the result.

### Method B: Paste Job Description
1. Copy text from an email offer, messaging conversation, or job board.
2. Open the extension popup, paste into the text box, and click **Check Pasted Text**.

### Method C: Check Screenshot (OCR)
1. Choose a PNG, JPG, or WebP image with the file picker.
2. Click **Send Screenshot for OCR** and confirm the destination. Tesseract OCR extracts the text and runs the assessment if the backend is available. Remote backends must use HTTPS; compare the extracted text with the image because OCR can miss or alter wording.

---

## 5. Permissions Rationale

The extension follows the principle of least privilege:

| Permission | Purpose |
| :--- | :--- |
| `activeTab` | Grants temporary access to the active browser tab only when the user explicitly clicks the extension or "Scan Current Page". |
| `scripting` | Executes a lightweight DOM script to extract text from job posting pages. |
| Local backend host access | Allows screenshot OCR requests to the local Flask server. |
| Optional HTTPS host access | Requested only for the specific remote backend origin selected by the user. |

Page scanning uses `activeTab` and `scripting` after the user presses the scan button; the extension does not scan automatically when its popup opens.

---

## 6. Model Weight Synchronization

Whenever you retrain the machine learning model in Python:

```powershell
python train_model.py
```

The training script automatically extracts the vocabulary, IDF table, logistic regression weights, and intercept, updating `const ML_MODEL` inside `extension/popup.js`. No manual copy-pasting is required.
