"""
Optional DistilBERT Sequence Classifier Upgrade (`train_distilbert.py`).

Fine-tunes `distilbert-base-uncased` (or a specified Hugging Face sequence
classification model) on the same fake job posting dataset using a three-way
stratified split (train / validation / held-out test), class-weighted cross-entropy
loss to handle severe class imbalance, and evaluates fraud-class precision, recall,
F1-score, and average precision.

Also compares DistilBERT's held-out test set metrics directly against the
TF-IDF + Logistic Regression baseline on the exact same held-out test split,
reporting empirically whether DistilBERT improves over the baseline.

Important notes:
- DistilBERT is an optional discriminative sequence classifier, NOT a generative
  LLM or explanation engine.
- All user-facing explanations in ScamShield AI remain 100% deterministic via
  `scamshield/explanation_service.py`.
- The core ScamShield AI application does NOT depend on PyTorch or Transformers.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from train_model import (
    DEFAULT_DATASET_PATH,
    TARGET_COLUMN,
    TEXT_COLUMNS,
    load_and_validate_dataset,
)

DEFAULT_DISTILBERT_DIR = os.environ.get("SCAMSHIELD_DISTILBERT_DIR", "models/distilbert_fake_job")
DEFAULT_COMPARISON_OUTPUT = os.environ.get(
    "SCAMSHIELD_DISTILBERT_REPORT", "models/distilbert_comparison.json"
)


def check_optional_dependencies() -> None:
    """Verify that torch, transformers, and scikit-learn are installed before training."""
    missing: List[str] = []
    for pkg in ("torch", "transformers", "sklearn"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append("scikit-learn" if pkg == "sklearn" else pkg)

    if missing:
        raise RuntimeError(
            "Optional DistilBERT training requires additional packages that are not installed: "
            f"{', '.join(missing)}.\n"
            "Install them via:\n"
            "    pip install torch transformers scikit-learn"
        )


def train_and_compare_distilbert(
    csv_path: Path,
    output_dir: Path,
    comparison_report_path: Path,
    pretrained_model_name: str = "distilbert-base-uncased",
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Fine-tune a DistilBERT sequence classifier with weighted loss, validate on a
    separate validation split, and compare with TF-IDF + Logistic Regression on
    a common held-out test split.
    """
    check_optional_dependencies()

    import torch  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import (  # type: ignore
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
    from torch.utils.data import Dataset  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    df = load_and_validate_dataset(csv_path, TEXT_COLUMNS, TARGET_COLUMN)
    texts = df["combined_text"].tolist()
    labels = df[TARGET_COLUMN].tolist()

    # Step 1: Hold out 20% test set (common to both TF-IDF baseline and DistilBERT)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=random_state,
        stratify=labels,
    )

    # Step 2: Split remaining 80% into train (85% of 80% = 68%) and validation (15% of 80% = 12%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.15,
        random_state=random_state,
        stratify=y_train_full,
    )

    # Step 3: Fit TF-IDF + Logistic Regression baseline on X_train for direct comparison on X_test
    baseline_pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=25000,
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
                    random_state=random_state,
                ),
            ),
        ]
    )
    baseline_pipeline.fit(X_train, y_train)
    base_pred = baseline_pipeline.predict(X_test)
    base_proba = baseline_pipeline.predict_proba(X_test)[:, 1]

    baseline_metrics = {
        "precision": round(float(precision_score(y_test, base_pred, pos_label=1, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, base_pred, pos_label=1, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, base_pred, pos_label=1, zero_division=0)), 4),
        "average_precision": round(float(average_precision_score(y_test, base_proba)), 4),
    }

    # Step 4: Prepare DistilBERT tokenizer and dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=2,
    ).to(device)

    class JobListingsDataset(Dataset):
        def __init__(self, item_texts: List[str], item_labels: List[int]) -> None:
            self.encodings = tokenizer(
                item_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.labels = torch.tensor(item_labels, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    train_dataset = JobListingsDataset(X_train, y_train)
    val_dataset = JobListingsDataset(X_val, y_val)
    test_dataset = JobListingsDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute inverse class frequency weights to handle class imbalance
    pos_count = max(1, sum(y_train))
    neg_count = max(1, len(y_train) - pos_count)
    total_train = len(y_train)
    weight_0 = total_train / (2.0 * neg_count)
    weight_1 = total_train / (2.0 * pos_count)
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def evaluate_loader(loader: Any) -> Dict[str, Any]:
        model.eval()
        all_preds: List[int] = []
        all_probs: List[float] = []
        all_targets: List[int] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[:, 1]
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())

        cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
        return {
            "precision": round(float(precision_score(all_targets, all_preds, pos_label=1, zero_division=0)), 4),
            "recall": round(float(recall_score(all_targets, all_preds, pos_label=1, zero_division=0)), 4),
            "f1_score": round(float(f1_score(all_targets, all_preds, pos_label=1, zero_division=0)), 4),
            "average_precision": round(float(average_precision_score(all_targets, all_probs)), 4),
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            },
        }

    validation_history: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        val_metrics = evaluate_loader(val_loader)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = round(epoch_loss / max(1, len(train_loader)), 4)
        validation_history.append(val_metrics)

    distilbert_test_metrics = evaluate_loader(test_loader)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    f1_diff = round(distilbert_test_metrics["f1_score"] - baseline_metrics["f1_score"], 4)
    if f1_diff > 0:
        comparison_summary = (
            f"On the common held-out test set, DistilBERT achieved a fraud-class F1-score of "
            f"{distilbert_test_metrics['f1_score']:.4f} vs. {baseline_metrics['f1_score']:.4f} "
            f"for the TF-IDF baseline (+{f1_diff:.4f})."
        )
    elif f1_diff < 0:
        comparison_summary = (
            f"On the common held-out test set, DistilBERT achieved a fraud-class F1-score of "
            f"{distilbert_test_metrics['f1_score']:.4f} vs. {baseline_metrics['f1_score']:.4f} "
            f"for the TF-IDF baseline ({f1_diff:.4f}). The TF-IDF baseline performed better on this split."
        )
    else:
        comparison_summary = (
            f"On the common held-out test set, both DistilBERT and the TF-IDF baseline achieved "
            f"an identical fraud-class F1-score of {distilbert_test_metrics['f1_score']:.4f}."
        )

    report: Dict[str, Any] = {
        "evaluated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pretrained_model": pretrained_model_name,
        "split_sizes": {
            "train": len(X_train),
            "validation": len(X_val),
            "held_out_test": len(X_test),
        },
        "tfidf_logreg_baseline_test_metrics": baseline_metrics,
        "distilbert_validation_history": validation_history,
        "distilbert_test_metrics": distilbert_test_metrics,
        "f1_difference_vs_baseline": f1_diff,
        "comparison_summary": comparison_summary,
        "note": (
            "DistilBERT is an optional discriminative sequence classifier. All user-facing "
            "explanations in ScamShield AI continue to be generated deterministically by "
            "scamshield/explanation_service.py."
        ),
    }

    comparison_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optional: Fine-tune DistilBERT and compare against the TF-IDF baseline."
    )
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the training CSV dataset (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=DEFAULT_DISTILBERT_DIR,
        help=f"Directory to save fine-tuned DistilBERT model and tokenizer (default: {DEFAULT_DISTILBERT_DIR}).",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=DEFAULT_COMPARISON_OUTPUT,
        help=f"Path to save JSON comparison report (default: {DEFAULT_COMPARISON_OUTPUT}).",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (default: 2).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = train_and_compare_distilbert(
            csv_path=Path(args.data_path),
            output_dir=Path(args.output_dir),
            comparison_report_path=Path(args.report_path),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    except FileNotFoundError:
        return 2
    except Exception as exc:
        print(f"[ScamShield AI] DistilBERT training halted: {exc}", file=sys.stderr)
        return 1

    print("================================================================================")
    print("[ScamShield AI] DistilBERT vs. TF-IDF Baseline Evaluation Complete")
    print("================================================================================")
    print(f"Baseline TF-IDF F1 : {report['tfidf_logreg_baseline_test_metrics']['f1_score']:.4f}")
    print(f"DistilBERT Test F1 : {report['distilbert_test_metrics']['f1_score']:.4f}")
    print(f"Summary            : {report['comparison_summary']}")
    print(f"Saved report to    : {args.report_path}")
    print("================================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
