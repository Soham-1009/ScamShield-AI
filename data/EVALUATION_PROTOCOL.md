# ScamShield Evaluation Data Protocol

Use this protocol before treating model or rule performance as evidence for deployment. The bundled EMSCAD-derived data is a research baseline, not a reviewed sample of current listings or a production validation set.

## Permitted collection

- Record the source, collection date, license or terms, and allowed use before retaining a listing.
- Do not bypass logins, CAPTCHAs, access controls, or platform restrictions. Do not collect applicant profiles, recruiter private messages, or personal contact details unless strictly necessary and specifically approved.
- Keep raw records access-controlled and local unless redistribution is explicitly permitted. Redact personal data before annotation and training; publish only approved derived data and aggregate metrics.
- Preserve a checksum, preparation script version, and dataset card for each snapshot. Never replace an older snapshot silently.

## Label guide

- **Scam (1):** There is credible evidence of deceptive recruitment or a known scam mechanism. A suspicious phrase by itself is not enough; record the evidence basis.
- **Reviewed non-scam (0):** The listing or employer was independently checked using reliable evidence. “No scam reported” or “looks normal” is not sufficient.
- **Uncertain:** Evidence is insufficient or reviewers disagree. Keep it out of binary training and report its count separately until adjudicated.
- Record the rule or evidence supporting a label, reviewer ID pseudonym, review date, and adjudication outcome. Have two reviewers label each evaluation example independently, then resolve disagreements before scoring.

## Grouping and splits

Use stable, pseudonymous `employer_group`, `source_group`, or `campaign_group` identifiers. Do not use a unique listing ID as a leakage-control group. Keep exact and near-duplicate listings, reposts, and campaign variants in one partition. Prefer a time-forward final evaluation set; also report performance by source, language, geography, and role type when sample sizes permit.

The training command must use group-isolated holdout and cross-validation by default. Random row splits are exploratory only. Record split IDs and the dataset checksum so the evaluation can be reproduced.

## Required report and release gate

Report confusion counts, false-positive rate, false-negative rate, precision, recall, PR-AUC, threshold-specific metrics, calibration diagnostics, and uncertainty intervals. Include subgroup counts and label disagreement rates. Calibration diagnostics do not make the model calibrated; fit and validate any calibrator on held-out representative data before describing outputs as probabilities.

Before model release, document the use case, acceptable false-positive and false-negative tradeoffs, minimum sample sizes, release thresholds, reviewer, model/data hashes, and rollback version. The project owner and domain reviewers must set numerical acceptance thresholds before examining the final holdout. If the evidence is too small or unrepresentative, mark evaluation **inconclusive** and keep the model experimental.

## Suggested annotation schema

`record_id`, `listing_text`, `label`, `label_status`, `evidence_basis`, `source_group`, `employer_group`, `campaign_group`, `language`, `country`, `collected_at`, `reviewer_a`, `reviewer_b`, `adjudication_status`, `dataset_version`.

The empty [`evaluation_template.csv`](evaluation_template.csv) records the required columns only; it is not training data. Keep completed files local unless publication and platform terms explicitly permit sharing.

Keep any mapping from pseudonymous IDs to real identities separate and access-controlled. Exclude personal email addresses, phone numbers, government IDs, bank details, applicant data, and tracking parameters from URLs.

## Reference

Use the [NIST AI RMF Measure guidance](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/) as a lifecycle reference for evaluation, uncertainty, and ongoing monitoring.
