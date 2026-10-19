"""
Deterministic explanation service for ScamShield AI.

Generates structured, human-readable explanations and safety recommendations
purely from predefined templates, rule-engine evidence, and metadata—without
using any LLM or generative AI service.

Strictly separates explicit matched evidence from general risk indicators,
avoids claiming that any company has been verified, and never declares a job
offer safe or legitimate solely because no warning rules triggered.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

BASELINE_VERIFICATION_RECOMMENDATIONS: List[str] = [
    (
        "Verify the listing on the employer's official corporate careers page rather than relying "
        "solely on third-party messages or forwarded social media posts."
    ),
    (
        "Confirm that recruiter emails come from an official corporate domain matching the company's "
        "verified website, not a free webmail address."
    ),
    (
        "Never pay upfront fees for registration, training, background checks, security deposits, "
        "or equipment delivery."
    ),
    (
        "Do not share bank login credentials, PINs, OTPs, or sensitive identity documents before "
        "completing a verifiable interview and confirming the employer's legitimacy."
    ),
]

UNCERTAINTY_DISCLAIMER: str = (
    "Important Disclaimer: ScamShield AI provides automated risk estimates based on configurable "
    "warning patterns and statistical text classification. An elevated risk assessment is not "
    "definitive proof of fraud, and the absence of detected warnings does not guarantee that a "
    "job or internship offer is legitimate. Always perform independent verification before sharing "
    "personal information or accepting an offer."
)


class DeterministicExplanationService:
    """
    Constructs structured, deterministic explanations from rule-engine outputs and ML metadata.
    """

    def generate_explanation(
        self,
        rule_evaluation: Dict[str, Any],
        ml_prediction: Optional[Dict[str, Any]] = None,
        input_word_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Build a consistent structured explanation object from detected evidence.
        """
        rule_assessment = rule_evaluation.get("rule_based_assessment", {})
        triggered_rules: List[Dict[str, Any]] = rule_evaluation.get("triggered_rules", [])
        risk_category = rule_assessment.get(
            "risk_category",
            "No Configured Warnings Detected (Verification Still Required)",
        )
        triggered_count = len(triggered_rules)

        # 1. Explicit evidence items (derived strictly from verbatim matched phrases)
        explicit_evidence: List[Dict[str, Any]] = []
        detailed_explanations: List[Dict[str, str]] = []
        recommendations: List[str] = []
        seen_recommendations = set()

        for rule in triggered_rules:
            explicit_evidence.append(
                {
                    "rule_id": rule["rule_id"],
                    "category": rule["category"],
                    "severity": rule["severity"],
                    "matched_phrase": rule["matched_phrase"],
                    "all_matched_phrases": rule.get("matched_phrases", [rule["matched_phrase"]]),
                    "evidence_type": "explicit_rule_match",
                }
            )
            detailed_explanations.append(
                {
                    "rule_id": rule["rule_id"],
                    "category": rule["category"],
                    "severity": rule["severity"],
                    "matched_phrase": rule["matched_phrase"],
                    "why_it_warrants_caution": (
                        f"Matched phrase \"{rule['matched_phrase']}\" under category "
                        f"'{rule['category']}': {rule['explanation']}"
                    ),
                    "recommended_verification_step": rule["verification_step"],
                }
            )
            step = rule["verification_step"]
            if step not in seen_recommendations:
                seen_recommendations.add(step)
                recommendations.append(step)

        # Append baseline verification steps so every response includes practical guidance
        for base_step in BASELINE_VERIFICATION_RECOMMENDATIONS:
            if base_step not in seen_recommendations:
                seen_recommendations.add(base_step)
                recommendations.append(base_step)

        # 2. General risk indicators (non-verbatim contextual observations, kept distinct from explicit evidence)
        general_risk_indicators: List[Dict[str, str]] = []

        if input_word_count > 0 and input_word_count < 25:
            general_risk_indicators.append(
                {
                    "indicator_type": "limited_listing_length",
                    "description": (
                        f"The submitted text contains only {input_word_count} word(s). Brief excerpts "
                        "or partial messages may omit important context or contractual terms."
                    ),
                }
            )

        if ml_prediction and ml_prediction.get("available"):
            ml_score_pct = ml_prediction.get("fraud_score_pct")
            ml_class = ml_prediction.get("predicted_class")
            general_risk_indicators.append(
                {
                    "indicator_type": "statistical_ml_comparison",
                    "description": (
                        f"Statistical TF-IDF + Logistic Regression classifier returned class "
                        f"'{ml_class}' with a fraud-class similarity score of {ml_score_pct}%. "
                        "This score reflects vocabulary similarity to historical dataset examples "
                        "and is not standalone evidence of fraud."
                    ),
                }
            )
        else:
            general_risk_indicators.append(
                {
                    "indicator_type": "ml_model_status",
                    "description": (
                        "Statistical ML model prediction is currently unavailable; this assessment "
                        "relies strictly on transparent rule-based pattern matching."
                    ),
                }
            )

        # 3. Deterministic summary generation using templates
        if triggered_count == 0:
            summary = (
                "No configured warning patterns were detected in the submitted text. "
                "However, the absence of matched warning rules does not confirm that the job or "
                "internship offer is legitimate or that the employer has been verified. "
                "Please follow the verification recommendations below before sharing personal data "
                "or making any commitments."
            )
        else:
            categories_found = list(dict.fromkeys(r["category"] for r in triggered_rules))
            categories_str = "; ".join(categories_found)
            summary = (
                f"Rule-based analysis assigned '{risk_category}' after detecting "
                f"{triggered_count} warning signal(s) across the following category(ies): "
                f"{categories_str}. These signals warrant careful scrutiny and independent "
                "verification, though automated pattern matches are risk estimates rather than "
                "definitive proof of fraud."
            )

        return {
            "summary": summary,
            "risk_category": risk_category,
            "triggered_warning_count": triggered_count,
            "explicit_evidence": explicit_evidence,
            "general_risk_indicators": general_risk_indicators,
            "detailed_explanations": detailed_explanations,
            "verification_recommendations": recommendations,
            "uncertainty_disclaimer": UNCERTAINTY_DISCLAIMER,
        }
