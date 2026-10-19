"""
Transparent, modular, and configurable rule-based risk engine for ScamShield AI.

Detects suspicious recruitment and internship signals across six core categories:
1. Upfront registration, training, or security fees
2. Requests for bank details, payment credentials, or sensitive personal information
3. Unrealistic salary and placement promises
4. Excessive urgency or pressure to accept an offer
5. Missing or unverifiable employer information
6. Requests to communicate through suspicious or unofficial channels

Every triggered rule extracts the exact verbatim phrase(s) from the input text,
pairs them with an understandable explanation and a concrete verification step,
and computes a transparent rule-based risk category without ever claiming that
the absence of warnings guarantees legitimacy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Sequence

# Strips standard job-portal anti-scam safety footers (e.g., Naukri.com's "Beware of imposters!" banner)
# so platform warning banners are not mistaken for employer fee demands.
PORTAL_SAFETY_DISCLAIMER_PATTERN = re.compile(
    r"(?:beware\s+of\s+imposters!.*?(?:read\s+more|\n\n|$)|"
    r"(?:\b(?:naukri\.com|internshala|indeed|linkedin|monster|foundit)\b|\bfraudsters\s+may\s+ask\b)"
    r"[^\n.]*?(?:registration\s+fee|refundable\s+fee|in\s+exchange\s+of\s+money)[^\n.]*\.?)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class RiskRule:
    """Configuration for a single recruitment warning rule."""

    rule_id: str
    category: str
    category_slug: str
    severity: str  # "high", "medium", "low"
    weight: int    # Contribution to rule score (0-100 scale)
    patterns: Sequence[str]
    explanation: str
    verification_step: str
    compiled_patterns: List[Pattern[str]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        compiled = [re.compile(pat, re.IGNORECASE) for pat in self.patterns]
        object.__setattr__(self, "compiled_patterns", compiled)


DEFAULT_RISK_RULES: List[RiskRule] = [
    # Category 1: Upfront registration, training, or security fees
    RiskRule(
        rule_id="RULE_UPFRONT_FEE",
        category="Upfront Registration, Training, or Security Fees",
        category_slug="upfront_fees",
        severity="high",
        weight=55,
        patterns=[
            r"\b(?:registration|onboarding|application|processing|training|certification|exam|security|verification|background\s+check|documentation|id\s+card|joining|administrative)\s+(?:fee|fees|charge|charges|deposit|payment|cost)\b",
            r"\b(?:refundable|one[-\s]?time|nominal|mandatory)\s+(?:security\s+)?(?:deposit|fee|charge|payment)\b",
            r"\bpay\s+(?:a\s+)?(?:small\s+|nominal\s+|refundable\s+)?(?:fee|deposit|amount|charge)\b",
            r"\b(?:fee|deposit|payment)\s+of\s+(?:rs\.?|inr|\$|usd|£|€)\s*\d[\d,]*(?:\.\d+)?\b",
            r"\b(?:rs\.?|inr|\$|usd|£|€)\s*\d[\d,]*(?:\.\d+)?\s+(?:for\s+|as\s+)?(?:registration|training|onboarding|security\s+deposit|processing|application\s+fee|laptop\s+deposit)\b",
        ],
        explanation=(
            "The listing references an upfront fee, deposit, or mandatory payment for registration, "
            "training, background verification, or onboarding. Legitimate employers generally pay "
            "candidates and do not require job seekers or interns to pay money to secure a role."
        ),
        verification_step=(
            "Do not send any money, wire transfers, UPI payments, or gift cards. Contact the company's "
            "human resources department through its official corporate website to verify whether any "
            "recruitment fee exists (legitimate hiring processes do not charge applicants)."
        ),
    ),
    RiskRule(
        rule_id="RULE_EQUIPMENT_OR_CHECK_SCHEME",
        category="Upfront Registration, Training, or Security Fees",
        category_slug="upfront_fees",
        severity="high",
        weight=50,
        patterns=[
            r"\b(?:laptop|equipment|software|hardware|macbook|workstation|starter\s+kit)\s+(?:security\s+)?(?:deposit|fee|shipping\s+fee|advance\s+payment)\b",
            r"\b(?:we\s+will\s+send\s+you\s+a\s+check|deposit\s+the\s+check|purchase\s+equipment\s+from\s+our\s+(?:approved\s+)?vendor)\b",
            r"\b(?:wire\s+transfer|crypto(?:currency)?|bitcoin|usdt|gift\s+card|itunes\s+card|steam\s+card|western\s+union)\s+(?:for\s+equipment|payment|to\s+complete\s+onboarding|to\s+our\s+vendor)\b",
        ],
        explanation=(
            "The offer mentions paying for company equipment, purchasing supplies from a designated "
            "vendor using a sent check, or using irreversible payment methods (wire, crypto, gift cards). "
            "Fake check and equipment-reimbursement schemes are common recruitment scams."
        ),
        verification_step=(
            "Never deposit a check from a prospective employer to buy equipment or wire funds back to a "
            "third-party vendor. Confirm how workplace equipment is provisioned directly through an "
            "in-person or verified corporate video interview."
        ),
    ),

    # Category 2: Requests for bank details, payment credentials, or sensitive personal information
    RiskRule(
        rule_id="RULE_BANK_OR_PAYMENT_CREDENTIALS",
        category="Requests for Bank Details, Payment Credentials, or Sensitive Personal Information",
        category_slug="sensitive_financial_or_personal_info",
        severity="high",
        weight=55,
        patterns=[
            r"\b(?:bank\s+account\s+(?:login|password|credentials|details|number)|online\s+banking\s+(?:login|password|id)|credit\s+card\s+(?:number|details|cvv)|debit\s+card\s+(?:number|pin|cvv)|upi\s+pin|atm\s+pin|one[-\s]?time\s+password|share\s+your\s+otp)\b",
            r"\b(?:provide|send|submit|share|email)\s+(?:your\s+)?(?:bank\s+account|routing\s+number|credit\s+card|debit\s+card|bank\s+statement)\s+(?:before\s+interview|to\s+apply|with\s+your\s+application|immediately)\b",
            r"\b(?:use\s+your\s+personal\s+bank\s+account|receive\s+and\s+transfer\s+funds|process\s+client\s+payments\s+through\s+your\s+account)\b",
        ],
        explanation=(
            "The text asks for sensitive financial credentials (such as banking logins, card numbers, "
            "PINs, or early bank details before formal hiring) or asks you to process financial "
            "transactions through your personal bank account (a potential money-mule arrangement)."
        ),
        verification_step=(
            "Never share banking passwords, PINs, OTPs, CVV codes, or allow your personal bank account "
            "to be used for company fund transfers. Payroll direct-deposit forms should only be completed "
            "after full verification, official interviews, and a signed contract via a verified HR portal."
        ),
    ),
    RiskRule(
        rule_id="RULE_PREMATURE_IDENTITY_DOCUMENTS",
        category="Requests for Bank Details, Payment Credentials, or Sensitive Personal Information",
        category_slug="sensitive_financial_or_personal_info",
        severity="medium",
        weight=35,
        patterns=[
            r"\b(?:send|email|whatsapp|submit|upload|provide)\s+(?:a\s+copy\s+of\s+|scans?\s+of\s+|your\s+)?(?:social\s+security\s+number|\bssn\b|aadhaar\s+card|pan\s+card|passport\s+copy|driver'?s?\s+license|tax\s+id|national\s+id)\s+(?:to\s+apply|before\s+interview|for\s+shortlisting|with\s+your\s+resume|immediately)\b",
        ],
        explanation=(
            "The listing requests sensitive government identification numbers or document scans "
            "at the initial application stage before any formal interview or employer verification."
        ),
        verification_step=(
            "Withhold government ID numbers (SSN, Aadhaar, PAN, passport scans) during initial job "
            "applications. Only provide identity verification documents after confirming the employer's "
            "legitimacy and reaching the official post-offer onboarding stage."
        ),
    ),

    # Category 3: Unrealistic salary and placement promises
    RiskRule(
        rule_id="RULE_UNREALISTIC_EARNINGS_PROMISE",
        category="Unrealistic Salary and Placement Promises",
        category_slug="unrealistic_salary_or_placement",
        severity="high",
        weight=45,
        patterns=[
            r"\bearn\s+(?:up\s+to\s+|over\s+)?(?:rs\.?|inr|\$|usd|£|€)\s*\d[\d,]*\s*(?:per\s+day|daily|per\s+hour|per\s+week|weekly)\s+(?:working\s+from\s+home|part[-\s]?time|with\s+no\s+experience|working\s+\d+\s*[-–to]*\s*\d*\s*hours?)\b",
            r"\b(?:work\s+(?:only\s+|just\s+)?\d+\s*(?:to\s+\d+\s*)?hours?\s+(?:a|per)\s+day\s+and\s+earn|guaranteed\s+(?:daily|weekly)\s+(?:income|payout|earnings)|make\s+\$\d[\d,]+\s+(?:a\s+week|per\s+week|a\s+day|daily)\s+(?:from\s+home|easily|guaranteed))\b",
            r"\b(?:high\s+salary\s+with\s+zero\s+experience|unlimited\s+earning\s+potential\s+no\s+skills\s+required|earn\s+lakhs\s+from\s+home\s+without\s+experience)\b",
        ],
        explanation=(
            "The advertisement promises unusually high daily or weekly earnings for minimal hours "
            "or requires no prior skills or qualifications. Disproportionate pay-to-effort claims "
            "are frequently used to attract students into task scams or advance-fee schemes."
        ),
        verification_step=(
            "Compare the stated compensation against standard entry-level or internship pay rates "
            "for the same role and region on established career platforms, and request a clear written "
            "breakdown of job duties and base salary structure."
        ),
    ),
    RiskRule(
        rule_id="RULE_GUARANTEED_PLACEMENT_NO_INTERVIEW",
        category="Unrealistic Salary and Placement Promises",
        category_slug="unrealistic_salary_or_placement",
        severity="medium",
        weight=35,
        patterns=[
            r"\b(?:100%\s*(?:job|placement|selection)\s*guarantee|guaranteed\s+(?:job\s+placement|internship\s+placement|offer\s+letter|hiring))\b",
            r"\b(?:no\s+interview\s+(?:required|needed)|direct\s+(?:joining|offer\s+letter|selection)\s+without\s+interview|instant\s+offer\s+letter|selected\s+without\s+(?:any\s+)?interview)\b",
        ],
        explanation=(
            "The offer promises guaranteed placement, 100% selection, or an immediate offer letter "
            "without conducting a technical or behavioral interview. Genuine employers evaluate "
            "candidates through structured interviews before extending job or internship offers."
        ),
        verification_step=(
            "Ask for the standard interview schedule and names of the hiring team members. Treat any "
            "unsolicited offer letter received without a real interview with strong caution."
        ),
    ),

    # Category 4: Excessive urgency or pressure to accept an offer
    RiskRule(
        rule_id="RULE_EXCESSIVE_URGENCY_PRESSURE",
        category="Excessive Urgency or Pressure to Accept",
        category_slug="excessive_urgency_or_pressure",
        severity="medium",
        weight=25,
        patterns=[
            r"\b(?:urgent\s+(?:requirement|hiring)\s*[-—:]?\s*(?:apply|join|confirm|respond)\s+(?:immediately|within\s+\d+\s*(?:hours?|minutes?)|today\s+only))\b",
            r"\b(?:confirm|accept|respond|pay|complete\s+registration)\s+within\s+(?:\d+|twenty[-\s]?four|24|48|two|few)\s*(?:hours?|minutes?|hrs?)\s+(?:or\s+(?:your\s+)?(?:offer|seat|slot)\s+will\s+be\s+cancelled|to\s+avoid\s+cancellation|to\s+secure\s+your\s+(?:seat|slot))\b",
            r"\b(?:only\s+\d+\s+(?:seats?|slots?|spots?)\s+left\s*[-—!,]?\s*(?:register|apply|pay|act)\s+(?:now|immediately|today)|act\s+immediately\s+before\s+slots?\s+fill|offer\s+expires\s+in\s+\d+\s*hours?)\b",
            r"\b(?:immediate\s+joining\s+today|sign\s+and\s+return\s+immediately\s+with\s+payment|last\s+chance\s+to\s+confirm\s+your\s+internship\s+slot)\b",
        ],
        explanation=(
            "The message uses high-pressure urgency tactics (such as strict hour-based deadlines, "
            "threats of immediate slot cancellation, or countdown pressure) designed to rush candidates "
            "into acting before verifying the organization."
        ),
        verification_step=(
            "Do not let artificial deadlines rush your due diligence. Legitimate employers provide "
            "reasonable time to review written offer terms and ask questions before making commitments."
        ),
    ),

    # Category 5: Missing or unverifiable employer information
    RiskRule(
        rule_id="RULE_UNVERIFIABLE_EMPLOYER_OR_FREE_EMAIL",
        category="Missing or Unverifiable Employer Information",
        category_slug="missing_or_unverifiable_employer_info",
        severity="medium",
        weight=30,
        patterns=[
            r"\b[\w.+-]+@(?:gmail|yahoo|outlook|hotmail|aol|protonmail|rediffmail|ymail|icloud|mail)\.(?:com|in|co\.uk|net|org)\b",
            r"\b(?:hiring\s+for\s+a\s+(?:reputed|leading|top|confidential|undisclosed)\s+(?:mnc|multinational|company|client)\s+(?:name\s+cannot\s+be\s+disclosed|company\s+name\s+confidential))\b",
            r"\b(?:company\s+name\s+(?:will\s+be\s+shared|disclosed)\s+after\s+(?:registration|payment|shortlisting)|undisclosed\s+employer)\b",
            r"\b(?:posted\s+by\s+freelancer|leading\s+management\s+consultant|india'?s\s+leading\s+mnc)\b",
        ],
        explanation=(
            "The listing uses a free personal webmail address (such as @gmail.com, @yahoo.com, or "
            "@outlook.com) or uses an anonymous placeholder ('Posted by Freelancer', 'Leading Management "
            "Consultant', 'India's leading MNC') without disclosing a verifiable company name."
        ),
        verification_step=(
            "Ask the recruiter for the registered corporate name, official website URL, and CIN/registration "
            "details before attending interviews or submitting documents, and verify the listing on the "
            "employer's official careers page."
        ),
    ),

    # Category 6: Requests to communicate through suspicious or unofficial channels
    RiskRule(
        rule_id="RULE_UNOFFICIAL_COMMUNICATION_CHANNEL",
        category="Requests to Communicate Through Suspicious or Unofficial Channels",
        category_slug="unofficial_communication_channels",
        severity="medium",
        weight=30,
        patterns=[
            r"\b(?:contact|message|ping|reach\s+out|apply|interview|connect|send\s+your\s+resume|dm)\s+(?:us\s+|me\s+|hr\s+|the\s+hiring\s+manager\s+)?(?:only\s+)?(?:on|via|through)\s+(?:whatsapp|telegram|signal|wechat|viber|google\s+hangouts)\b",
            r"\b(?:whatsapp|telegram)\s+(?:only|number|hr|recruiter|interview|chat)\s*(?:at|on|:)?\s*(?:\+?\d[\d\s-]{7,15}|@[a-zA-Z0-9_]{4,}|t\.me/[a-zA-Z0-9_]+)?\b",
            r"\bt\.me/[a-zA-Z0-9_]+\b",
            r"\b(?:text[-\s]?only\s+interview|interview\s+via\s+(?:telegram|whatsapp|text\s+chat))\b",
        ],
        explanation=(
            "The listing directs applicants to conduct interviews, submit applications, or communicate "
            "primarily through encrypted messaging apps (such as WhatsApp, Telegram, or Signal) or "
            "text-only chat rather than verifiable corporate email and career portals."
        ),
        verification_step=(
            "Request that all official interview invites and offer documents be sent from a verified "
            "corporate email address, and cross-check the recruiter's identity on the company's official website."
        ),
    ),
]


class RiskRuleEngine:
    """
    Evaluates recruitment text against a configurable set of transparent risk rules.
    """

    def __init__(self, rules: Optional[Sequence[RiskRule]] = None) -> None:
        self.rules: List[RiskRule] = list(rules) if rules is not None else list(DEFAULT_RISK_RULES)

    def evaluate(self, text: str) -> Dict[str, Any]:
        """
        Run all configured rules against `text` and return structured evidence and risk category.
        """
        if not isinstance(text, str):
            text = ""

        # Strip boilerplate job-portal safety disclaimers (e.g. Naukri's "Beware of imposters!" footer)
        # so they are not mistaken for upfront fee demands by the employer.
        cleaned_text = PORTAL_SAFETY_DISCLAIMER_PATTERN.sub(" ", text).strip()
        word_count = len(cleaned_text.split()) if cleaned_text else 0

        triggered_rules: List[Dict[str, Any]] = []
        all_matched_phrases: List[str] = []
        raw_score = 0
        has_high_severity = False

        for rule in self.rules:
            rule_matches: List[str] = []
            seen_lower = set()

            for compiled_pat in rule.compiled_patterns:
                for match in compiled_pat.finditer(cleaned_text):
                    exact_phrase = match.group(0).strip()
                    if exact_phrase and exact_phrase.lower() not in seen_lower:
                        seen_lower.add(exact_phrase.lower())
                        rule_matches.append(exact_phrase)

            if rule_matches:
                raw_score += rule.weight
                if rule.severity == "high":
                    has_high_severity = True

                for phrase in rule_matches:
                    if phrase not in all_matched_phrases:
                        all_matched_phrases.append(phrase)

                triggered_rules.append(
                    {
                        "rule_id": rule.rule_id,
                        "category": rule.category,
                        "category_slug": rule.category_slug,
                        "severity": rule.severity,
                        "weight": rule.weight,
                        "matched_phrase": rule_matches[0],
                        "matched_phrases": rule_matches,
                        "explanation": rule.explanation,
                        "verification_step": rule.verification_step,
                    }
                )

        rule_score = min(100, raw_score)
        triggered_count = len(triggered_rules)

        # Determine risk category strictly from rule-based evidence.
        # Never show High Risk without explicit supporting rule evidence.
        # Never label an offer "Legitimate" or "Safe" simply because 0 rules matched.
        if triggered_count == 0:
            if word_count < 20:
                risk_category = "Inconclusive — Limited Text Context"
                risk_level_code = "inconclusive"
                assessment_note = (
                    "No configured warning rules matched, but the submitted text is very brief "
                    f"({word_count} words). Absence of warning triggers does not confirm legitimacy; "
                    "provide the full job listing and verify the employer independently."
                )
            else:
                risk_category = "No Configured Warnings Detected (Verification Still Required)"
                risk_level_code = "no_rules_triggered"
                assessment_note = (
                    "No configured rule-based warning patterns were detected in this text. "
                    "However, the absence of matched warning rules does not guarantee that an offer "
                    "is legitimate. Always verify the employer independently."
                )
        elif has_high_severity or rule_score >= 50:
            risk_category = "High Risk (Multiple or Critical Warning Signals Detected)"
            risk_level_code = "high"
            assessment_note = (
                f"Detected {triggered_count} warning rule(s) with high-severity recruitment risk "
                "indicators. Review the matched phrases and verification steps carefully before proceeding."
            )
        elif rule_score >= 25:
            risk_category = "Medium Risk (Caution Warranted — Warning Signals Detected)"
            risk_level_code = "medium"
            assessment_note = (
                f"Detected {triggered_count} warning signal(s) that warrant closer scrutiny and "
                "independent employer verification before sharing personal details or accepting."
            )
        else:
            risk_category = "Low-to-Moderate Caution (Minor Warning Indicator Detected)"
            risk_level_code = "low_caution"
            assessment_note = (
                f"Detected {triggered_count} minor warning indicator(s). Verify the recruitment "
                "process through official corporate channels."
            )

        return {
            "rule_based_assessment": {
                "risk_category": risk_category,
                "risk_level_code": risk_level_code,
                "rule_score": rule_score,
                "triggered_rule_count": triggered_count,
                "total_rules_checked": len(self.rules),
                "note": assessment_note,
            },
            "triggered_rules": triggered_rules,
            "exact_matched_phrases": all_matched_phrases,
        }


def compute_experimental_triage_score(
    rule_assessment: Dict[str, Any],
    ml_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute a clearly documented experimental combined triage score.
    """
    rule_score = float(rule_assessment.get("rule_score", 0))
    triggered_count = int(rule_assessment.get("triggered_rule_count", 0))
    ml_available = bool(ml_result.get("available", False))
    ml_prob = ml_result.get("fraud_probability_score")

    formula_doc = (
        "experimental_triage_score = round(0.65 * rule_score + 0.35 * (ml_fraud_probability * 100), 1) "
        "when ML model is available; capped at 'Review Needed (Statistical Signal Only)' if "
        "triggered_rule_count == 0 so high risk is never assigned without explicit rule evidence."
    )

    if not ml_available or ml_prob is None:
        return {
            "available": False,
            "score": None,
            "label": "Unavailable (Requires Trained ML Model)",
            "formula": formula_doc,
            "weights": {"rule_engine": 0.65, "ml_classifier": 0.35},
            "is_experimental": True,
            "is_calibrated_probability": False,
            "explanation": (
                "Experimental combined triage score is not computed because no trained ML model is "
                "currently loaded. Refer directly to the Rule-Based Assessment above."
            ),
        }

    ml_score_100 = float(ml_prob) * 100.0
    combined = round(0.65 * rule_score + 0.35 * ml_score_100, 1)

    if triggered_count == 0:
        if ml_score_100 >= 50.0:
            label = "Review Needed — Statistical ML Signal Only (No Rule Evidence Matched)"
        else:
            label = "No Rule Triggers / Low Statistical Signal (Independent Verification Required)"
    else:
        if combined >= 50.0 or rule_assessment.get("risk_level_code") == "high":
            label = "Elevated Triage Priority (Supported by Rule Evidence)"
        elif combined >= 25.0:
            label = "Moderate Triage Priority (Review Matched Evidence)"
        else:
            label = "Standard Verification Recommended"

    return {
        "available": True,
        "score": combined,
        "label": label,
        "formula": formula_doc,
        "components": {
            "rule_score_0_to_100": rule_score,
            "ml_fraud_score_0_to_100": round(ml_score_100, 1),
        },
        "weights": {"rule_engine": 0.65, "ml_classifier": 0.35},
        "is_experimental": True,
        "is_calibrated_probability": False,
        "explanation": (
            "This experimental triage index combines the rule-based risk score (65% weight) and "
            "the statistical ML fraud-class score (35% weight) for prioritization only. "
            "It is not a calibrated probability of fraud."
        ),
    }
