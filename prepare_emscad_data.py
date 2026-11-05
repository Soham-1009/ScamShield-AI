"""Prepare the locally downloaded EMSCAD balanced subset for ScamShield training.

This creates a PII-reduced CSV plus a pseudonymous evaluation group. The group is
derived from repeated company_profile text (not a verified employer ID); rows
without a substantial profile get a unique row group and therefore no employer
isolation guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import pandas as pd

TEXT_COLUMNS = ("title", "company_profile", "description", "requirements", "benefits")
DEFAULT_INPUT = Path("data/fake_job_postings_emscad_balanced_en.csv")
DEFAULT_OUTPUT = Path("data/fake_job_postings_emscad_prepared.csv")
DEFAULT_MANIFEST = Path("data/fake_job_postings_emscad_prepared.manifest.json")

EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"
)


def prepare_dataset(input_path: Path, output_path: Path, manifest_path: Path) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    df = pd.read_csv(input_path)
    required = {"job_id", "company_profile", "fraudulent", *TEXT_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if df["job_id"].isna().any() or df["job_id"].astype(str).str.strip().eq("").any():
        raise ValueError("job_id must be present to create a stable fallback group for blank profiles.")

    normalized_profiles = (
        df["company_profile"].fillna("").astype(str).str.casefold().str.split().str.join(" ")
    )
    profile_groups = []
    row_fallbacks = 0
    for profile, job_id in zip(normalized_profiles, df["job_id"].astype(str)):
        if len(profile) >= 40:
            digest = hashlib.sha256(profile.encode("utf-8")).hexdigest()
            profile_groups.append(f"profile:{digest}")
        else:
            digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
            profile_groups.append(f"row:{digest}")
            row_fallbacks += 1

    redaction_counts = {column: {"emails": 0, "phone_like": 0} for column in TEXT_COLUMNS}
    for column in TEXT_COLUMNS:
        series = df[column].fillna("").astype(str)
        redaction_counts[column]["emails"] = int(series.str.count(EMAIL_PATTERN).sum())
        series = series.str.replace(EMAIL_PATTERN, " [EMAIL] ", regex=True)
        redaction_counts[column]["phone_like"] = int(series.str.count(PHONE_PATTERN).sum())
        df[column] = series.str.replace(PHONE_PATTERN, " [PHONE] ", regex=True)

    df["evaluation_group"] = profile_groups
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest = {
        "source_file": input_path.name,
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "prepared_file": output_path.name,
        "prepared_sha256": output_hash,
        "rows": len(df),
        "label_counts": {str(k): int(v) for k, v in df["fraudulent"].value_counts().sort_index().items()},
        "group_column": "evaluation_group",
        "group_method": "SHA-256 of normalized company_profile when length >= 40; otherwise unique job_id hash",
        "rows_using_row_fallback": row_fallbacks,
        "unique_profile_groups": len({g for g in profile_groups if g.startswith("profile:")}),
        "pii_pattern_redactions": redaction_counts,
        "limitations": [
            "company_profile-derived groups are a proxy and are not verified employer IDs",
            "rows with blank or short company profiles use unique row groups and are not employer-isolated",
            "source subset is balanced and historically collected; it does not represent current scam prevalence",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = prepare_dataset(args.input, args.output, args.manifest)
    except Exception as exc:
        raise SystemExit(f"Dataset preparation failed: {exc}") from exc
    print(f"Prepared rows: {manifest['rows']}")
    print(f"Profile groups: {manifest['unique_profile_groups']}")
    print(f"Rows using unique-row fallback: {manifest['rows_using_row_fallback']}")
    print(f"Prepared CSV: {args.output}")
    print(f"Manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
