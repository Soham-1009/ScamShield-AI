"""
Machine learning inference wrapper for ScamShield AI (`scamshield/ml_detector.py`).

Loads the fitted Scikit-learn TF-IDF + Logistic Regression pipeline persisted via
Joblib (`models/fake_job_model.joblib`) along with its evaluation metadata
(`models/model_metadata.json`).

Key guarantees:
- Never fabricates predictions, probabilities, or model metadata when the model
  artifact or required ML libraries (`joblib`, `scikit-learn`) are missing.
- Clearly states that statistical ML predictions are not proof of fraud or legitimacy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_MODEL_PATH = os.environ.get("SCAMSHIELD_MODEL_PATH", "models/fake_job_model.joblib")
DEFAULT_METADATA_PATH = os.environ.get("SCAMSHIELD_MODEL_METADATA_PATH", "models/model_metadata.json")

ML_PREDICTION_DISCLAIMER = (
    "Statistical ML predictions measure text similarity against historical training data and "
    "are NOT proof that a specific job offer is fraudulent or legitimate. Always review "
    "explicit rule-based evidence and verify the employer independently."
)


class MLDetector:
    """
    Manages loading and inference for the baseline TF-IDF + Logistic Regression classifier.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.metadata_path = Path(metadata_path or DEFAULT_METADATA_PATH)
        self._pipeline: Any = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._load_error: Optional[str] = None
        self.reload()

    def reload(self) -> bool:
        """
        Attempt to load the trained model pipeline and metadata from disk.

        Returns True if a valid model was loaded, False otherwise.
        """
        self._pipeline = None
        self._metadata = None
        self._load_error = None

        if not self.model_path.exists() or not self.model_path.is_file():
            self._load_error = (
                f"Trained model file not found at '{self.model_path}'. "
                "Run 'python train_model.py --data-path data/fake_job_postings.csv' to train and save the model."
            )
            return False

        try:
            import joblib  # type: ignore
        except ImportError:
            self._load_error = (
                "Required dependency 'joblib' (and/or 'scikit-learn') is not installed. "
                "Install dependencies via 'pip install -r requirements.txt'."
            )
            return False

        try:
            loaded_model = joblib.load(self.model_path)
            if not hasattr(loaded_model, "predict"):
                self._load_error = (
                    f"Artifact at '{self.model_path}' does not implement a valid 'predict' method."
                )
                return False
            self._pipeline = loaded_model
        except Exception as exc:
            self._load_error = f"Failed to load model from '{self.model_path}': {exc}"
            return False

        if self.metadata_path.exists() and self.metadata_path.is_file():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        self._metadata = data
            except Exception:
                self._metadata = None

        return True

    @property
    def is_available(self) -> bool:
        """Return True if a trained classification pipeline is loaded and ready."""
        return self._pipeline is not None

    def get_health_status(self) -> Dict[str, Any]:
        """Return structured health information for GET /api/health."""
        model_version = (
            self._metadata.get("model_version", "1.0.0-tfidf-logreg")
            if isinstance(self._metadata, dict)
            else ("1.0.0-tfidf-logreg" if self.is_available else None)
        )
        return {
            "available": self.is_available,
            "model_type": "TF-IDF + Logistic Regression (scikit-learn)" if self.is_available else None,
            "model_path": str(self.model_path),
            "model_version": model_version,
            "status_message": (
                "Trained ML classifier loaded and ready."
                if self.is_available
                else (self._load_error or "ML model unavailable.")
            ),
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run inference on `text` using the loaded Scikit-learn pipeline.

        If no trained model is available, explicitly reports `available: False`
        and `fraud_probability_score: None` rather than returning fabricated or zero scores.
        """
        if not self.is_available or self._pipeline is None:
            return {
                "available": False,
                "predicted_label": None,
                "predicted_class": None,
                "fraud_probability_score": None,
                "fraud_score_pct": None,
                "model_version": None,
                "metadata": None,
                "disclaimer": ML_PREDICTION_DISCLAIMER,
                "status_message": (
                    self._load_error
                    or "ML predictions are unavailable because no trained model is loaded."
                ),
            }

        try:
            raw_pred = self._pipeline.predict([text])[0]
            predicted_label = int(raw_pred)
            predicted_class = (
                "fraudulent_pattern_detected"
                if predicted_label == 1
                else "low_statistical_fraud_similarity"
            )

            fraud_prob: Optional[float] = None
            fraud_pct: Optional[float] = None
            if hasattr(self._pipeline, "predict_proba"):
                proba_row = self._pipeline.predict_proba([text])[0]
                classes = list(getattr(self._pipeline, "classes_", [0, 1]))
                if 1 in classes:
                    fraud_idx = classes.index(1)
                    fraud_prob = round(float(proba_row[fraud_idx]), 4)
                    fraud_pct = round(fraud_prob * 100.0, 1)
                elif len(proba_row) >= 2:
                    fraud_prob = round(float(proba_row[1]), 4)
                    fraud_pct = round(fraud_prob * 100.0, 1)

            model_version = "1.0.0-tfidf-logreg"
            public_metadata: Dict[str, Any] = {
                "architecture": "TF-IDF Vectorizer + Logistic Regression (class_weight='balanced')",
            }
            if isinstance(self._metadata, dict):
                model_version = str(self._metadata.get("model_version", model_version))
                for key in ("model_version", "trained_at_utc", "threshold", "dataset_summary"):
                    if key in self._metadata:
                        public_metadata[key] = self._metadata[key]

            return {
                "available": True,
                "predicted_label": predicted_label,
                "predicted_class": predicted_class,
                "fraud_probability_score": fraud_prob,
                "fraud_score_pct": fraud_pct,
                "model_version": model_version,
                "metadata": public_metadata,
                "disclaimer": ML_PREDICTION_DISCLAIMER,
                "status_message": "ML inference completed. Prediction is a statistical estimate, not proof of fraud.",
            }
        except Exception as exc:
            return {
                "available": False,
                "predicted_label": None,
                "predicted_class": None,
                "fraud_probability_score": None,
                "fraud_score_pct": None,
                "model_version": None,
                "metadata": None,
                "disclaimer": ML_PREDICTION_DISCLAIMER,
                "status_message": f"ML model inference encountered an error: {exc}",
            }
