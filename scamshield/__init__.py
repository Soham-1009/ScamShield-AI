"""
ScamShield AI — Explainable Fake Job & Internship Detector.

Core package providing:
- Modular, configurable rule-based recruitment risk detection (`risk_rules`)
- Deterministic, template-driven explanation generation (`explanation_service`)
- Scikit-learn TF-IDF + Logistic Regression model inference (`ml_detector`)
- Tesseract OCR text extraction for offer screenshots (`ocr_service`)
- Strict API input and file upload validation (`validation`)
"""

__version__ = "1.0.0"
__all__ = [
    "__version__",
    "risk_rules",
    "explanation_service",
    "ml_detector",
    "ocr_service",
    "validation",
]
