"""
Unit tests for the transparent RiskRuleEngine, experimental triage scoring,
and DeterministicExplanationService (`tests/test_rules.py`).

Synthetic examples in this file are for functional testing of rule matching
and deterministic template behavior only; they are NOT evidence of ML model accuracy.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "ScamShieldAI"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scamshield.explanation_service import DeterministicExplanationService
from scamshield.risk_rules import (
    RiskRule,
    RiskRuleEngine,
    compute_experimental_triage_score,
)


class TestRiskRulesAndExplanations(unittest.TestCase):
    """Functional tests for rule detection, exact evidence extraction, and deterministic explanations."""

    def setUp(self) -> None:
        self.engine = RiskRuleEngine()
        self.explainer = DeterministicExplanationService()

    def test_upfront_fee_detection(self) -> None:
        """Synthetic case 1: A job advertisement requesting an upfront fee."""
        text = (
            "Junior Data Analyst Internship. Selected candidates must pay a refundable "
            "registration fee of $150 and a training fee before onboarding starts."
        )
        result = self.engine.evaluate(text)
        assessment = result["rule_based_assessment"]
        rule_ids = [r["rule_id"] for r in result["triggered_rules"]]

        self.assertIn("RULE_UPFRONT_FEE", rule_ids)
        self.assertEqual(assessment["risk_level_code"], "high")
        self.assertGreaterEqual(assessment["rule_score"], 50)
        self.assertTrue(
            any(
                "registration fee" in phrase.lower() or "training fee" in phrase.lower()
                for phrase in result["exact_matched_phrases"]
            )
        )

        # Verify deterministic explanation
        explanation = self.explainer.generate_explanation(result, ml_prediction=None, input_word_count=22)
        self.assertEqual(explanation["triggered_warning_count"], len(result["triggered_rules"]))
        self.assertGreaterEqual(len(explanation["explicit_evidence"]), 1)
        self.assertIn("not definitive proof of fraud", explanation["uncertainty_disclaimer"])

    def test_unrealistic_salary_promise(self) -> None:
        """Synthetic case 2: A job offer making an unrealistic salary promise."""
        text = (
            "Remote Part-Time Data Entry Specialist! Work only 2 hours a day and earn $4500 a week "
            "easily from home. Guaranteed daily payout and 100% placement guarantee with no interview required."
        )
        result = self.engine.evaluate(text)
        rule_ids = [r["rule_id"] for r in result["triggered_rules"]]

        self.assertIn("RULE_UNREALISTIC_EARNINGS_PROMISE", rule_ids)
        self.assertIn("RULE_GUARANTEED_PLACEMENT_NO_INTERVIEW", rule_ids)
        self.assertEqual(result["rule_based_assessment"]["risk_level_code"], "high")

    def test_legitimate_looking_job_description_not_labeled_guaranteed_safe(self) -> None:
        """
        Synthetic case 3: A legitimate-looking job description with no obvious warning signs.
        Must trigger 0 rules AND must NOT claim the offer is verified or guaranteed legitimate.
        """
        text = (
            "Software Engineering Intern (Summer 2027). Responsibilities include collaborating "
            "with backend engineers to build REST APIs in Python, writing unit tests with pytest, "
            "and participating in code reviews. Qualifications: pursuing a BS in Computer Science "
            "or equivalent practical experience with data structures and relational databases. "
            "Selection process includes an online coding assessment followed by two technical "
            "video interviews with the engineering team."
        )
        result = self.engine.evaluate(text)
        assessment = result["rule_based_assessment"]

        self.assertEqual(assessment["triggered_rule_count"], 0)
        self.assertEqual(assessment["rule_score"], 0)
        self.assertEqual(assessment["risk_level_code"], "no_rules_triggered")
        self.assertIn("Verification Still Required", assessment["risk_category"])
        self.assertNotIn("Legitimate", assessment["risk_category"])

        explanation = self.explainer.generate_explanation(
            result,
            ml_prediction={"available": False},
            input_word_count=len(text.split()),
        )
        self.assertEqual(explanation["explicit_evidence"], [])
        self.assertIn("no configured warning patterns were detected", explanation["summary"].lower())
        self.assertIn("does not confirm", explanation["summary"].lower())
        self.assertGreaterEqual(len(explanation["verification_recommendations"]), 4)

    def test_urgency_language_without_payment_request(self) -> None:
        """
        Synthetic case 4: Urgency language without a payment request.
        Should trigger the urgency rule ('medium' severity) without triggering upfront fee rules.
        """
        text = (
            "Marketing Operations Intern. Urgent hiring: confirm within 24 hours or your offer "
            "will be cancelled. Only 2 slots left, act immediately before slots fill for our "
            "upcoming cohort."
        )
        result = self.engine.evaluate(text)
        rule_ids = [r["rule_id"] for r in result["triggered_rules"]]

        self.assertIn("RULE_EXCESSIVE_URGENCY_PRESSURE", rule_ids)
        self.assertNotIn("RULE_UPFRONT_FEE", rule_ids)
        self.assertNotIn("RULE_EQUIPMENT_OR_CHECK_SCHEME", rule_ids)
        self.assertEqual(result["rule_based_assessment"]["risk_level_code"], "medium")

    def test_multiple_warning_rules_triggered_by_genuine_evidence(self) -> None:
        """
        Synthetic case 11: Multiple warning rules triggered simultaneously across categories.
        """
        text = (
            "Hiring for a confidential MNC (company name will be shared after registration). "
            "No interview required — direct offer letter! Contact HR only on WhatsApp or "
            "Telegram at t.me/quickjobs_hr or email recruiter.jobs99@gmail.com. "
            "To confirm within 24 hours to secure your seat, pay a refundable security deposit "
            "of Rs. 2500 and share your bank account login credentials."
        )
        result = self.engine.evaluate(text)
        rule_ids = {r["rule_id"] for r in result["triggered_rules"]}

        self.assertIn("RULE_UPFRONT_FEE", rule_ids)
        self.assertIn("RULE_BANK_OR_PAYMENT_CREDENTIALS", rule_ids)
        self.assertIn("RULE_GUARANTEED_PLACEMENT_NO_INTERVIEW", rule_ids)
        self.assertIn("RULE_EXCESSIVE_URGENCY_PRESSURE", rule_ids)
        self.assertIn("RULE_UNVERIFIABLE_EMPLOYER_OR_FREE_EMAIL", rule_ids)
        self.assertIn("RULE_UNOFFICIAL_COMMUNICATION_CHANNEL", rule_ids)
        self.assertEqual(result["rule_based_assessment"]["risk_level_code"], "high")
        self.assertEqual(result["rule_based_assessment"]["rule_score"], 100)

        for rule in result["triggered_rules"]:
            self.assertTrue(rule["matched_phrase"])
            self.assertIn(rule["matched_phrase"].lower(), text.lower())
            self.assertTrue(rule["explanation"])
            self.assertTrue(rule["verification_step"])

    def test_custom_configurable_rule_injection(self) -> None:
        """Verify that RiskRuleEngine accepts custom modular rules."""
        custom_rule = RiskRule(
            rule_id="CUSTOM_CRYPTO_TASK",
            category="Custom Task Scam Pattern",
            category_slug="custom_task_scam",
            severity="high",
            weight=60,
            patterns=[r"\bboost\s+app\s+ratings\s+with\s+usdt\b"],
            explanation="Detected app-rating recharge task scam pattern.",
            verification_step="Never deposit cryptocurrency to unlock commission tasks.",
        )
        custom_engine = RiskRuleEngine(rules=[custom_rule])
        res = custom_engine.evaluate("Join our team to boost app ratings with USDT daily.")
        self.assertEqual(len(res["triggered_rules"]), 1)
        self.assertEqual(res["triggered_rules"][0]["rule_id"], "CUSTOM_CRYPTO_TASK")
        self.assertEqual(res["triggered_rules"][0]["matched_phrase"].lower(), "boost app ratings with usdt")

    def test_experimental_triage_score_guardrails(self) -> None:
        """
        Verify experimental triage score calculation:
        1. Returns available=False when ML model is unavailable.
        2. Never labels High Risk when triggered_rule_count == 0 even if ML score is high.
        """
        no_rule_assessment = {
            "risk_category": "No Configured Warnings Detected (Verification Still Required)",
            "risk_level_code": "no_rules_triggered",
            "rule_score": 0,
            "triggered_rule_count": 0,
        }
        ml_unavailable = {"available": False, "fraud_probability_score": None}
        triage_missing = compute_experimental_triage_score(no_rule_assessment, ml_unavailable)
        self.assertFalse(triage_missing["available"])
        self.assertIsNone(triage_missing["score"])
        self.assertTrue(triage_missing["is_experimental"])
        self.assertFalse(triage_missing["is_calibrated_probability"])

        # High ML probability but 0 rule triggers -> must NOT claim High Risk
        ml_high = {"available": True, "fraud_probability_score": 0.92}
        triage_stat_only = compute_experimental_triage_score(no_rule_assessment, ml_high)
        self.assertTrue(triage_stat_only["available"])
        self.assertIn("Statistical ML Signal Only", triage_stat_only["label"])
        self.assertNotIn("High Risk", triage_stat_only["label"])


if __name__ == "__main__":
    unittest.main()
