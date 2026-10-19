"""
Baseline Machine Learning Training Pipeline (`train_model.py`).

Trains a TF-IDF + Logistic Regression binary classifier on the public Fake Job
Postings dataset (`fake_job_postings.csv`).

Key features:
- Configurable dataset path via `--data-path` or `SCAMSHIELD_DATASET_PATH`.
- Displays clear setup instructions and exits cleanly if the dataset or
  required ML dependencies (`scikit-learn`, `joblib`) are missing.
- Never fabricates a trained model, accuracy, or evaluation metrics.
- Combines relevant text columns (`title`, `company_profile`, `description`,
  `requirements`, `benefits`) while safely handling missing values.
- Uses a stratified train/test split (`test_size=0.2`, `random_state=42`).
- Prevents data leakage by fitting `TfidfVectorizer` strictly inside a
  Scikit-learn `Pipeline` on the training split only; the held-out test split
  is never used for feature fitting or threshold tuning.
- Evaluates fraud-class precision, recall, F1-score, confusion matrix, and
  average precision (PR-AUC).
- Saves the fitted pipeline to `models/fake_job_model.joblib` and evaluation
  metadata to `models/model_metadata.json`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

TEXT_COLUMNS: List[str] = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]
TARGET_COLUMN: str = "fraudulent"

DEFAULT_DATASET_PATH = os.environ.get("SCAMSHIELD_DATASET_PATH", "data/fake_job_postings.csv")
DEFAULT_MODEL_OUTPUT = os.environ.get("SCAMSHIELD_MODEL_PATH", "models/fake_job_model.joblib")
DEFAULT_METADATA_OUTPUT = os.environ.get(
    "SCAMSHIELD_MODEL_METADATA_PATH", "models/model_metadata.json"
)


def print_dataset_setup_instructions(dataset_path: Path) -> None:
    """Print clear instructions when the training dataset is missing."""
    msg = f"""
================================================================================
[ScamShield AI] Training Dataset Not Found: {dataset_path}
================================================================================
To train the baseline TF-IDF + Logistic Regression model, supply a valid CSV
dataset containing recruitment listings (such as the public Employment Scam
Aegean Dataset / 'fake_job_postings.csv').

1. Expected CSV Columns:
   - Required target column: 'fraudulent' (0 for non-fraudulent, 1 for fraudulent)
   - Text columns (at least 'description' or a subset of):
     'title', 'company_profile', 'description', 'requirements', 'benefits'

2. Setup Steps:
   - Place the CSV file at:
       {dataset_path}
   - Or specify a custom path when running this script:
       python train_model.py --data-path path/to/fake_job_postings.csv

No model artifact or evaluation metrics were generated because the dataset is
missing. ScamShield AI never fabricates trained models or metrics.
================================================================================
"""
    print(msg.strip(), file=sys.stderr)


def load_and_validate_dataset(
    csv_path: Path,
    text_columns: Sequence[str] = TEXT_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Load the dataset CSV, validate required columns and class labels, and build
    the combined text input column (`combined_text`) without mutating label distributions.
    """
    if not csv_path.exists() or not csv_path.is_file():
        print_dataset_setup_instructions(csv_path)
        raise FileNotFoundError(f"Dataset file not found at '{csv_path}'.")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset at '{csv_path}' is empty (0 rows).")

    if target_column not in df.columns:
        raise ValueError(
            f"Dataset is missing required target column '{target_column}'. "
            f"Found columns: {list(df.columns)}"
        )

    present_text_cols = [col for col in text_columns if col in df.columns]
    if not present_text_cols:
        raise ValueError(
            f"Dataset must include at least one of the expected text columns {list(text_columns)}. "
            f"Found columns: {list(df.columns)}"
        )

    # Drop rows where target label is missing
    df = df.dropna(subset=[target_column]).copy()
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df[df[target_column].isin([0, 1])].copy()
    df[target_column] = df[target_column].astype(int)

    if df.empty:
        raise ValueError(
            f"Dataset at '{csv_path}' has no valid rows with binary '{target_column}' labels (0 or 1)."
        )

    class_counts = df[target_column].value_counts().to_dict()
    if 0 not in class_counts or 1 not in class_counts:
        raise ValueError(
            f"Dataset must contain both class 0 and class 1 in '{target_column}'. "
            f"Found class counts: {class_counts}"
        )

    if min(class_counts.values()) < 2:
        raise ValueError(
            "Each class (0 and 1) must have at least 2 samples to perform a stratified train/test split."
        )

    # Handle missing text fields safely and combine relevant columns into one input string
    for col in present_text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["combined_text"] = df[present_text_cols].agg(" ".join, axis=1).str.replace(
        r"\s+", " ", regex=True
    ).str.strip()

    # Filter out rows where all text columns were completely empty
    df = df[df["combined_text"].str.len() > 0].copy()
    if df.empty:
        raise ValueError("All rows had empty text fields after combining text columns.")

    return df


def train_and_evaluate(
    csv_path: Path,
    model_output_path: Path,
    metadata_output_path: Path,
    test_size: float = 0.20,
    random_state: int = 42,
    max_features: int = 25000,
) -> Dict[str, Any]:
    """
    Execute the end-to-end training and evaluation pipeline without data leakage.
    """
    # Validate dataset existence and schema first so missing-dataset guidance is shown immediately
    df = load_and_validate_dataset(csv_path)

    try:
        import joblib  # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.metrics import (  # type: ignore
            average_precision_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
        from sklearn.model_selection import train_test_split  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Required machine learning dependencies ('scikit-learn' and 'joblib') are not installed.\n"
            "Install them using:\n"
            "    pip install -r requirements.txt"
        ) from exc

    X = df["combined_text"].tolist()
    y = df[TARGET_COLUMN].tolist()

    # Stratified split BEFORE any TF-IDF fitting to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                    min_df=2 if len(X_train) >= 20 else 1,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    C=1.0,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )

    # Fit strictly on X_train, y_train
    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set (using default 0.50 decision threshold; test set is NEVER used for tuning)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    precision = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    avg_precision = float(average_precision_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["non_fraudulent (0)", "fraudulent (1)"],
        output_dict=True,
        zero_division=0,
    )

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_output_path)

    metadata: Dict[str, Any] = {
        "model_version": "1.0.0-tfidf-logreg",
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "architecture": "Pipeline(TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True), LogisticRegression(class_weight='balanced'))",
        "threshold": 0.50,
        "data_leakage_prevention": (
            "Stratified train/test split performed prior to TfidfVectorizer fitting. "
            "Held-out test set was used strictly for final reporting and never for threshold tuning."
        ),
        "dataset_summary": {
            "source_path": str(csv_path),
            "total_valid_rows": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "fraudulent_count_total": int(sum(y)),
            "non_fraudulent_count_total": int(len(y) - sum(y)),
            "fraud_rate_pct": round((sum(y) / len(y)) * 100.0, 2),
            "text_columns_combined": [c for c in TEXT_COLUMNS if c in df.columns],
        },
        "test_set_evaluation": {
            "fraudulent_class_precision": round(precision, 4),
            "fraudulent_class_recall": round(recall, 4),
            "fraudulent_class_f1_score": round(f1, 4),
            "fraudulent_class_average_precision_pr_auc": round(avg_precision, 4),
            "confusion_matrix": {
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
                "matrix_2x2": [[tn, fp], [fn, tp]],
            },
            "classification_report": report_dict,
        },
        "disclaimer": (
            "Model predictions reflect statistical word/phrase associations in historical training "
            "data and do not constitute definitive proof that a job advertisement is fraudulent or legitimate."
        ),
    }

    with open(metadata_output_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the ScamShield AI TF-IDF + Logistic Regression baseline model."
    )
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the training CSV dataset (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default=DEFAULT_MODEL_OUTPUT,
        help=f"Destination path for the fitted Joblib pipeline (default: {DEFAULT_MODEL_OUTPUT}).",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=DEFAULT_METADATA_OUTPUT,
        help=f"Destination path for JSON evaluation metadata (default: {DEFAULT_METADATA_OUTPUT}).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proportion of dataset reserved for stratified test split (default: 0.20).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splitting and training (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    csv_path = Path(args.data_path)
    model_path = Path(args.model_path)
    metadata_path = Path(args.metadata_path)

    try:
        metadata = train_and_evaluate(
            csv_path=csv_path,
            model_output_path=model_path,
            metadata_output_path=metadata_path,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    except FileNotFoundError:
        return 2
    except Exception as exc:
        print(f"[ScamShield AI] Training halted: {exc}", file=sys.stderr)
        return 1

    eval_metrics = metadata["test_set_evaluation"]
    cm = eval_metrics["confusion_matrix"]
    print("================================================================================")
    print("[ScamShield AI] Baseline TF-IDF + Logistic Regression Training Complete")
    print("================================================================================")
    print(f"Dataset rows (valid): {metadata['dataset_summary']['total_valid_rows']}")
    print(f"Train / Test split  : {metadata['dataset_summary']['train_rows']} / {metadata['dataset_summary']['test_rows']}")
    print(f"Fraudulent Precision: {eval_metrics['fraudulent_class_precision']:.4f}")
    print(f"Fraudulent Recall   : {eval_metrics['fraudulent_class_recall']:.4f}")
    print(f"Fraudulent F1-Score : {eval_metrics['fraudulent_class_f1_score']:.4f}")
    print(f"Average Precision   : {eval_metrics['fraudulent_class_average_precision_pr_auc']:.4f}")
    print(
        f"Confusion Matrix    : TN={cm['true_negatives']}, FP={cm['false_positives']}, "
        f"FN={cm['false_negatives']}, TP={cm['true_positives']}"
    )
    print(f"Saved model to      : {model_path}")
    print(f"Saved metadata to   : {metadata_path}")
    print("================================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
