# ScamShieldAI Dataset Card

Copy this file to `data/DATASET_CARD.md` for every training dataset. Do not commit listing text or personal data unless its license, collection terms, and privacy review explicitly permit that release.

## Dataset identity

- Name and version:
- Maintainer:
- Collection period:
- Source and source license / terms:
- Dataset checksum (SHA-256):
- Storage location and access controls:

## Intended use and limits

- Intended use: research and evaluation of employment scam warning signals.
- Not intended for: automated hiring decisions, employer blacklists, definitive fraud findings, or guarantees of safety.
- Known coverage gaps (region, language, job type, channel, time period):
- Known label and collection bias:

## Labels and review

- Target column: `fraudulent` (`0` = reviewed non-fraud example, `1` = reviewed fraud example).
- Label policy and evidence standard:
- Reviewer process and adjudication for disagreements:
- Label review date:

Do not infer a fraud label solely from suspicious wording. Record the basis for each label in an access-controlled annotation file and retain only the minimum evidence needed to audit it.

## Grouping and split fields

Provide a stable, non-identifying `source_group`, `employer_group`, or `campaign_group` column for group-isolated evaluation. The value should be a pseudonymous grouping ID reused for related records; a unique row ID such as `job_id` is not a useful leakage-control group.

- Group column used for evaluation:
- Group definition:
- Are groups reused across labels? Explain:
- Any known duplicate / near-duplicate campaigns:

Do not include recruiter names, email addresses, phone numbers, account identifiers, CVs, government IDs, bank details, private chat text, or source URLs that contain user-specific tokens. Redact or exclude those fields before training.

## Reproduction

- Preparation script and version:
- Labeling protocol version:
- Train command and arguments:
- Evaluation split strategy:
- Random seed:
- Model / dependency versions:
- Output report location:

Store raw source material only where collection terms allow it, limit access and retention, and keep the dataset checksum and preparation notes so authorized users can reproduce the derived dataset without publishing restricted records.
