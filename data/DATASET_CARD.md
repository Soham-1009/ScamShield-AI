# EMSCAD Balanced English Subset (Local Training Copy)

## Identity and provenance

- Dataset: `gplsi/fake_job_postings_balanced_en`, pinned to repository revision `1973102aeabe205188908bd6f94585242a801bef`.
- Dataset card: <https://huggingface.co/datasets/gplsi/fake_job_postings_balanced_en/tree/1973102aeabe205188908bd6f94585242a801bef>.
- Upstream source: Fake Job Posting Prediction / EMSCAD, attributed by the dataset card to Kaggle. The published study describes the underlying Employment Scam Aegean Dataset (EMSCAD): Vidros et al., 2017, <https://doi.org/10.3390/fi9010006>.
- License shown for this redistributed dataset: Creative Commons Attribution 4.0 International (CC BY 4.0). Attribute GPLSI UA and link this dataset card when redistributing derived artifacts. The card separately reports the upstream Kaggle dataset as CC0; preserve both provenance statements and review terms before any public redistribution.
- Retrieval date: 2026-11-05.
- Raw source file: `fake_job_postings_emscad_balanced_en.csv` (kept local and ignored by Git).
- Raw SHA-256: `0df4b46d0c5d1ac8272aaa10e480141d8e02f57196ae57db6041bacb074792c4`.
- Prepared file: `fake_job_postings_emscad_prepared.csv` (kept local and ignored by Git).
- Prepared SHA-256: `83658bb2c7508e86915044273a96d4653ac5ea94d90ab5da5fb9335f1fdac78e`.
- Preparation manifest: `data/fake_job_postings_emscad_prepared.manifest.json`.

## Observed data and preparation

The dataset card describes 1,730 records split 865/865. The pinned CSV downloaded for this run contains 1,732 rows split 866/866. This discrepancy is recorded rather than silently dropping or adding rows. The training pipeline later drops exact duplicate combined text and rejects identical text with conflicting labels.

`prepare_emscad_data.py` hashes normalized `company_profile` text of at least 40 characters into the pseudonymous `evaluation_group`; shorter or blank profiles use a unique hash of `job_id`. The prepared source has 431 reusable profile groups and 720 unique-row fallbacks. This is only a repeated-profile proxy, not verified employer or campaign identity. It cannot guarantee group isolation for fallback rows.

The preparer redacts email and phone-like patterns from the five text fields used by the classifier (`title`, `company_profile`, `description`, `requirements`, `benefits`) and records only counts. Pattern-based redaction is imperfect; the source and prepared CSV remain local and should not be published without a privacy review.

## Labels, intended use, and limitations

- Target: `fraudulent` (`0` = non-fraudulent, `1` = fraudulent), as provided by the source dataset. These are inherited labels, not newly reviewed or adjudicated by ScamShieldAI.
- Intended use: research and prototype evaluation of employment-listing warning signals.
- Not intended for: definitive fraud determinations, employer blacklists, employment decisions, or claims that a listing is safe.
- The source is a balanced historical subset of the EMSCAD corpus, not a representative sample of current listings or real-world fraud prevalence. The classifier's scores must not be interpreted as fraud probabilities.
- The source card reports only balancing as a modification. Preparation adds PII-pattern redaction and a pseudonymous evaluation group; it does not independently validate source labels.

## Reproduction

From the repository root, with project dependencies installed:

```powershell
python prepare_emscad_data.py
python train_model.py --data-path data\fake_job_postings_emscad_prepared.csv --group-column evaluation_group --model-path models\fake_job_model.joblib --metadata-path models\model_metadata.json
```

Training uses a group-isolated holdout and grouped cross-validation. Because row fallbacks are unique, the evaluation is only partially protected against employer leakage and must be treated as a research diagnostic, not deployment evidence. The model metadata records the prepared data checksum, split properties, threshold metrics, and calibration diagnostics.
