"""
Baseline Machine Learning Training Pipeline (`train_model.py`).

Trains a regularized TF-IDF + Logistic Regression binary classifier on the public Fake Job
Postings dataset (`fake_job_postings.csv`).

Key features:
- Configurable dataset path via `--data-path` or `SCAMSHIELD_DATASET_PATH`.
- Displays clear setup instructions and exits cleanly if the dataset or
  required ML dependencies (`scikit-learn`, `joblib`) are missing.
- Never fabricates a trained model, accuracy, or evaluation metrics.
- Deduplicates text records and strips synthetic batch tags to prevent train/test leakage.
- Combines relevant text columns (`title`, `company_profile`, `description`,
  `requirements`, `benefits`) while safely handling missing values.
- Uses a stratified train/test split (`test_size=0.2`, `random_state=42`).
- Prevents data leakage by fitting `TfidfVectorizer` strictly inside a
  Scikit-learn `Pipeline` on the training split only; the held-out test split
  is never used for feature fitting or threshold tuning.
- Applies 5-fold Stratified Cross-Validation on the training split to select
  an optimal regularized inverse penalty `C`.
- Evaluates both train-set and test-set performance, reporting precision, recall,
  F1-score, confusion matrix, PR-AUC, and the generalization gap (`train_f1 - test_f1`).
- Saves the fitted pipeline to `models/fake_job_model.joblib` and evaluation
  metadata to `models/model_metadata.json`.
- Synchronizes model weights with the Chrome extension (`extension/popup.js`) for
  offline in-browser inference.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
DEFAULT_POPUP_JS_PATH = "extension/popup.js"

BATCH_TAG_REGEX = re.compile(r"\s*\(Batch\s*\d+\)", re.IGNORECASE)


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
    Load the dataset CSV, validate required columns and class labels, sanitize
    synthetic batch tags, deduplicate rows to avoid train/test leakage, and build
    the combined text input column (`combined_text`).
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

    # Handle missing text fields safely and strip synthetic batch tags
    for col in present_text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()
        df[col] = df[col].apply(lambda text: BATCH_TAG_REGEX.sub("", text).strip())

    df["combined_text"] = (
        df[present_text_cols]
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Filter out rows where all text columns were completely empty
    df = df[df["combined_text"].str.len() > 0].copy()
    if df.empty:
        raise ValueError("All rows had empty text fields after combining text columns.")

    # Deduplicate on combined_text to guarantee zero data leakage between train and test
    initial_count = len(df)
    df = df.drop_duplicates(subset=["combined_text"]).copy()
    dropped_duplicates = initial_count - len(df)
    if dropped_duplicates > 0:
        print(
            f"[ScamShield AI] Notice: Dropped {dropped_duplicates} duplicate job listing(s) to prevent train/test leakage."
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

    return df


def export_model_to_extension_js(
    pipeline: Any,
    popup_js_path: Path,
) -> bool:
    """
    Extract vocabulary, IDF weights, logistic regression coefficients, and intercept
    from the fitted pipeline and update `const ML_MODEL` in `extension/popup.js`.
    """
    if not popup_js_path.exists() or not popup_js_path.is_file():
        return False

    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    stop_words = sorted(list(tfidf.get_stop_words() or []))
    vocab = tfidf.vocabulary_
    idf_arr = tfidf.idf_
    coef_arr = clf.coef_[0]
    intercept_val = round(float(clf.intercept_[0]), 4)

    # Build token -> idf and token -> coef mappings
    idf_dict: Dict[str, float] = {}
    coef_dict: Dict[str, float] = {}
    for token, idx in vocab.items():
        idf_dict[token] = round(float(idf_arr[idx]), 4)
        coef_dict[token] = round(float(coef_arr[idx]), 4)

    ml_model_obj = {
        "stop": stop_words,
        "idf": idf_dict,
        "coef": coef_dict,
        "intercept": intercept_val,
    }

    try:
        content = popup_js_path.read_text(encoding="utf-8")
        replacement = f"const ML_MODEL = {json.dumps(ml_model_obj)};"
        new_content = re.sub(
            r"const ML_MODEL\s*=\s*\{.*?\};",
            replacement,
            content,
            count=1,
            flags=re.DOTALL,
        )
        if new_content != content:
            popup_js_path.write_text(new_content, encoding="utf-8")
            print(f"[ScamShield AI] Synced {len(vocab)} model weights to extension at {popup_js_path}")
            return True
    except Exception as exc:
        print(f"[ScamShield AI] Warning: Failed to sync model to extension: {exc}", file=sys.stderr)

    return False


def train_and_evaluate(
    csv_path: Path,
    model_output_path: Path,
    metadata_output_path: Path,
    test_size: float = 0.20,
    random_state: int = 42,
    max_features: int = 5000,
    c_candidates: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 1.0),
    popup_js_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute the end-to-end training, hyperparameter regularization tuning,
    and generalization diagnostic pipeline without data leakage.
    """
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
        from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # type: ignore
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

    # Stratified K-Fold cross-validation on X_train to select regularized C with optimal probability calibration
    n_splits = min(5, min(y_train.count(0), y_train.count(1)))
    best_c = 0.5
    best_cv_loss = float("inf")
    cv_diagnostics: Dict[str, Any] = {}

    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for c_val in c_candidates:
            candidate_pipe = Pipeline(
                steps=[
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=max_features,
                            ngram_range=(1, 2),
                            stop_words="english",
                            sublinear_tf=True,
                            min_df=2 if len(X_train) >= 20 else 1,
                            max_df=0.95,
                        ),
                    ),
                    (
                        "clf",
                        LogisticRegression(
                            C=c_val,
                            class_weight="balanced",
                            max_iter=1000,
                            solver="lbfgs",
                            random_state=random_state,
                        ),
                    ),
                ]
            )
            scores_f1 = cross_val_score(candidate_pipe, X_train, y_train, cv=cv, scoring="f1")
            scores_loss = -cross_val_score(
                candidate_pipe, X_train, y_train, cv=cv, scoring="neg_log_loss"
            )
            mean_f1 = float(scores_f1.mean())
            std_f1 = float(scores_f1.std())
            mean_loss = float(scores_loss.mean())
            cv_diagnostics[f"C_{c_val}"] = {
                "mean_f1": round(mean_f1, 4),
                "std_f1": round(std_f1, 4),
                "mean_log_loss": round(mean_loss, 4),
            }
            # Select candidate that minimizes log-loss while maintaining top F1
            if mean_f1 >= 0.90 and mean_loss < best_cv_loss:
                best_cv_loss = mean_loss
                best_c = c_val
    else:
        best_c = 0.5

    # Final pipeline fitted strictly on X_train, y_train with optimal regularized C
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
                    max_df=0.95,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    C=best_c,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    # 1. Train-set evaluation (to detect potential overfitting and compute generalization gap)
    y_train_pred = pipeline.predict(X_train)
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    train_precision = float(precision_score(y_train, y_train_pred, pos_label=1, zero_division=0))
    train_recall = float(recall_score(y_train, y_train_pred, pos_label=1, zero_division=0))
    train_f1 = float(f1_score(y_train, y_train_pred, pos_label=1, zero_division=0))
    train_avg_precision = float(average_precision_score(y_train, y_train_proba))
    train_cm = confusion_matrix(y_train, y_train_pred, labels=[0, 1])

    # 2. Test-set evaluation (held-out split, never used for fitting or threshold tuning)
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_precision = float(precision_score(y_test, y_test_pred, pos_label=1, zero_division=0))
    test_recall = float(recall_score(y_test, y_test_pred, pos_label=1, zero_division=0))
    test_f1 = float(f1_score(y_test, y_test_pred, pos_label=1, zero_division=0))
    test_avg_precision = float(average_precision_score(y_test, y_test_proba))
    test_cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    test_tn, test_fp, test_fn, test_tp = (
        int(test_cm[0, 0]),
        int(test_cm[0, 1]),
        int(test_cm[1, 0]),
        int(test_cm[1, 1]),
    )
    test_report_dict = classification_report(
        y_test,
        y_test_pred,
        labels=[0, 1],
        target_names=["non_fraudulent (0)", "fraudulent (1)"],
        output_dict=True,
        zero_division=0,
    )

    generalization_gap_f1 = round(train_f1 - test_f1, 4)

    # Extract top predictive features for transparency and auditability
    vocab = pipeline.named_steps["tfidf"].get_feature_names_out()
    coefs = pipeline.named_steps["clf"].coef_[0]
    sorted_features = sorted(zip(vocab, coefs), key=lambda x: x[1])
    top_legitimate = [
        {"feature": feat, "weight": round(float(w), 4)} for feat, w in sorted_features[:15]
    ]
    top_fraudulent = [
        {"feature": feat, "weight": round(float(w), 4)} for feat, w in sorted_features[-15:][::-1]
    ]

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_output_path)

    metadata: Dict[str, Any] = {
        "model_version": "1.1.0-tfidf-regularized-logreg",
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "architecture": (
            f"Pipeline(TfidfVectorizer(max_features={max_features}, ngram_range=(1,2), sublinear_tf=True), "
            f"LogisticRegression(C={best_c}, class_weight='balanced'))"
        ),
        "hyperparameters": {
            "max_features": max_features,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "selected_regularization_C": best_c,
            "candidate_C_values": list(c_candidates),
            "cross_validation_folds": n_splits,
        },
        "threshold": 0.50,
        "data_leakage_prevention": (
            "Deduplication performed on combined text to eliminate duplicate records. "
            "Synthetic batch tags stripped prior to splitting. "
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
        "cross_validation_results": cv_diagnostics,
        "train_set_evaluation": {
            "fraudulent_class_precision": round(train_precision, 4),
            "fraudulent_class_recall": round(train_recall, 4),
            "fraudulent_class_f1_score": round(train_f1, 4),
            "fraudulent_class_average_precision_pr_auc": round(train_avg_precision, 4),
            "confusion_matrix": {
                "true_negatives": int(train_cm[0, 0]),
                "false_positives": int(train_cm[0, 1]),
                "false_negatives": int(train_cm[1, 0]),
                "true_positives": int(train_cm[1, 1]),
            },
        },
        "test_set_evaluation": {
            "fraudulent_class_precision": round(test_precision, 4),
            "fraudulent_class_recall": round(test_recall, 4),
            "fraudulent_class_f1_score": round(test_f1, 4),
            "fraudulent_class_average_precision_pr_auc": round(test_avg_precision, 4),
            "confusion_matrix": {
                "true_negatives": test_tn,
                "false_positives": test_fp,
                "false_negatives": test_fn,
                "true_positives": test_tp,
                "matrix_2x2": [[test_tn, test_fp], [test_fn, test_tp]],
            },
            "classification_report": test_report_dict,
        },
        "generalization_diagnostics": {
            "generalization_gap_f1": generalization_gap_f1,
            "status": (
                "Healthy generalization — train/test metrics closely aligned."
                if abs(generalization_gap_f1) <= 0.08
                else "Noticeable generalization gap — inspect feature distributions."
            ),
        },
        "top_features": {
            "legitimate_predictors": top_legitimate,
            "fraudulent_predictors": top_fraudulent,
        },
        "disclaimer": (
            "Model predictions reflect statistical word/phrase associations in historical training "
            "data and do not constitute definitive proof that a job advertisement is fraudulent or legitimate."
        ),
    }

    with open(metadata_output_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    # Sync to Chrome extension if path provided or default exists
    target_js_path = popup_js_path or Path(DEFAULT_POPUP_JS_PATH)
    if target_js_path.exists():
        export_model_to_extension_js(pipeline, target_js_path)

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
        "--popup-js-path",
        type=str,
        default=DEFAULT_POPUP_JS_PATH,
        help=f"Destination path for Chrome extension popup.js (default: {DEFAULT_POPUP_JS_PATH}).",
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
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum vocabulary features for TfidfVectorizer (default: 5000).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    csv_path = Path(args.data_path)
    model_path = Path(args.model_path)
    metadata_path = Path(args.metadata_path)
    popup_js_path = Path(args.popup_js_path)

    try:
        metadata = train_and_evaluate(
            csv_path=csv_path,
            model_output_path=model_path,
            metadata_output_path=metadata_path,
            test_size=args.test_size,
            random_state=args.random_state,
            max_features=args.max_features,
            popup_js_path=popup_js_path,
        )
    except FileNotFoundError:
        return 2
    except Exception as exc:
        print(f"[ScamShield AI] Training halted: {exc}", file=sys.stderr)
        return 1

    eval_metrics = metadata["test_set_evaluation"]
    train_metrics = metadata["train_set_evaluation"]
    cm = eval_metrics["confusion_matrix"]
    gen_diag = metadata["generalization_diagnostics"]
    print("================================================================================")
    print("[ScamShield AI] Baseline TF-IDF + Logistic Regression Training Complete")
    print("================================================================================")
    print(f"Dataset rows (valid): {metadata['dataset_summary']['total_valid_rows']}")
    print(
        f"Train / Test split  : {metadata['dataset_summary']['train_rows']} / {metadata['dataset_summary']['test_rows']}"
    )
    print(f"Optimal Regularized C: {metadata['hyperparameters']['selected_regularization_C']}")
    print(
        f"Train F1 / Test F1  : {train_metrics['fraudulent_class_f1_score']:.4f} / {eval_metrics['fraudulent_class_f1_score']:.4f}"
    )
    print(f"Generalization Gap  : {gen_diag['generalization_gap_f1']:+.4f} ({gen_diag['status']})")
    print(f"Test Precision      : {eval_metrics['fraudulent_class_precision']:.4f}")
    print(f"Test Recall         : {eval_metrics['fraudulent_class_recall']:.4f}")
    print(f"Test PR-AUC         : {eval_metrics['fraudulent_class_average_precision_pr_auc']:.4f}")
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
