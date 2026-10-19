/**
 * ScamShield AI — Main-Screen Browser Extension (`popup.js`).
 *
 * - All controls (Scan Page, Job Text box, Check Pasted Text, and Screenshot Check)
 *   are visible directly on the main screen without any hidden dropdowns.
 * - Runs both the Rule Engine and the trained TF-IDF + Logistic Regression ML model
 *   100% inside Chrome with zero CORS or network errors.
 */

(function () {
  "use strict";

  const API_BASE_URL = "http://127.0.0.1:5000";
  const ML_MODEL = {"stop": ["all", "and", "any", "are", "can", "for", "from", "has", "have", "intern", "internship", "into", "job", "not", "our", "out", "per", "role", "that", "the", "this", "use", "using", "via", "who", "will", "with", "work", "working", "you", "your"], "idf": {"batch": 1.0, "of": 1.5769, "to": 1.6795, "in": 1.9232, "or": 1.9232, "python": 1.9232, "company": 2.0716, "experience": 2.2459, "fee": 2.2459, "home": 2.2459, "offer": 2.2459, "pay": 2.2459, "stipend": 2.2459, "fee of": 2.4572, "guaranteed": 2.4572, "hours": 2.4572, "instant": 2.4572, "interview": 2.4572, "no": 2.4572, "on": 2.4572, "only": 2.4572, "paid": 2.4572, "placement": 2.4572, "week": 2.4572, "backend": 2.7255, "basic": 2.7255, "before": 2.7255, "build": 2.7255, "card": 2.7255, "code": 2.7255, "data": 2.7255, "day": 2.7255, "developer": 2.7255, "earn": 2.7255, "engineering": 2.7255, "familiarity": 2.7255, "health": 2.7255, "hiring": 2.7255, "immediately": 2.7255, "instant offer": 2.7255, "javascript": 2.7255, "joining": 2.7255, "learning": 2.7255, "letter": 2.7255, "needed": 2.7255, "of rs": 2.7255, "offer letter": 2.7255, "refundable": 2.7255, "required": 2.7255, "rs": 2.7255, "salary": 2.7255, "skills": 2.7255, "software": 2.7255, "sql": 2.7255, "strong": 2.7255, "structured": 2.7255, "technical": 2.7255, "test": 2.7255, "time": 2.7255, "today": 2.7255, "web": 2.7255, "without": 2.7255, "write": 2.7255, "account": 3.0932, "after": 3.0932, "api": 3.0932, "apis": 3.0932, "at": 3.0932, "automated": 3.0932, "bank": 3.0932, "bank account": 3.0932, "batch confidential": 3.0932, "be": 3.0932, "benefits": 3.0932, "building": 3.0932, "candidates": 3.0932, "check": 3.0932, "classification": 3.0932, "client": 3.0932, "cloud": 3.0932, "company building": 3.0932, "computer": 3.0932, "confidential": 3.0932, "copy": 3.0932, "daily": 3.0932, "dashboards": 3.0932, "direct": 3.0932, "documentation": 3.0932, "earn rs": 3.0932, "enterprise": 3.0932, "equipment": 3.0932, "flask": 3.0932, "fund": 3.0932, "git": 3.0932, "guarantee": 3.0932, "home guaranteed": 3.0932, "hybrid": 3.0932, "india": 3.0932, "interview required": 3.0932, "it": 3.0932, "junior": 3.0932, "laptop": 3.0932, "leading": 3.0932, "learn": 3.0932, "make": 3.0932, "make week": 3.0932, "mentorship": 3.0932, "mnc": 3.0932, "modern": 3.0932, "modern javascript": 3.0932, "monthly": 3.0932, "monthly stipend": 3.0932, "of web": 3.0932, "onboarding": 3.0932, "or interview": 3.0932, "part": 3.0932, "part time": 3.0932, "pay refundable": 3.0932, "pipelines": 3.0932, "portal": 3.0932, "processing": 3.0932, "product": 3.0932, "pune": 3.0932, "queries": 3.0932, "recruiter": 3.0932, "remote": 3.0932, "rest": 3.0932, "rest apis": 3.0932, "science": 3.0932, "scikit": 3.0932, "scikit learn": 3.0932, "security": 3.0932, "selected": 3.0932, "send": 3.0932, "slot": 3.0932, "slot no": 3.0932, "slots": 3.0932, "tech": 3.0932, "telegram": 3.0932, "tests": 3.0932, "tools": 3.0932, "trainee": 3.0932, "training": 3.0932, "transfer": 3.0932, "typing": 3.0932, "understanding": 3.0932, "understanding of": 3.0932, "undisclosed": 3.0932, "urgent": 3.0932, "weekly": 3.0932, "whatsapp": 3.0932, "within": 3.0932, "within hours": 3.0932, "write automated": 3.0932, "aadhaar": 3.681, "aadhaar card": 3.681, "ability": 3.681, "ability to": 3.681, "account login": 3.681, "account to": 3.681, "across": 3.681, "across major": 3.681, "act": 3.681, "act immediately": 3.681, "administrative": 3.681, "administrative assistant": 3.681, "adoption": 3.681, "adoption metrics": 3.681, "after first": 3.681, "after registration": 3.681, "agency": 3.681, "agency personal": 3.681, "ai": 3.681, "ai data": 3.681, "analyst": 3.681, "analyst batch": 3.681, "analytical": 3.681, "analytical communication": 3.681, "analytics": 3.681, "analytics consulting": 3.681}, "coef": {"batch": -0.101, "of": 0.112, "to": 0.4504, "in": -0.8519, "or": 0.3692, "python": -0.8361, "company": -0.2964, "experience": 0.0343, "fee": 0.6212, "home": 0.6928, "offer": 0.3022, "pay": 0.5318, "stipend": -0.5388, "fee of": 0.4407, "guaranteed": 0.4752, "hours": 0.4491, "instant": 0.4228, "interview": 0.5912, "no": 0.5409, "on": 0.1183, "only": 0.5553, "paid": -0.554, "placement": 0.2206, "week": 0.2751, "backend": -0.3667, "basic": -0.1473, "before": 0.4349, "build": -0.3657, "card": 0.4797, "code": -0.333, "data": 0.0519, "day": 0.1114, "developer": -0.4733, "earn": 0.3768, "engineering": -0.342, "familiarity": -0.3683, "health": -0.4035, "hiring": 0.5186, "immediately": 0.4484, "instant offer": 0.3416, "javascript": -0.4695, "joining": 0.3623, "learning": -0.4352, "letter": 0.3416, "needed": 0.434, "of rs": 0.3968, "offer letter": 0.3416, "refundable": 0.3823, "required": 0.4098, "rs": 0.5633, "salary": 0.1851, "skills": -0.1218, "software": -0.3635, "sql": -0.4563, "strong": -0.4265, "structured": -0.3683, "technical": 0.1433, "test": -0.2377, "time": 0.4484, "today": 0.3623, "web": -0.4665, "without": 0.4142, "write": -0.3692, "account": 0.3646, "after": 0.2894, "api": -0.4074, "apis": -0.2254, "at": -0.0732, "automated": -0.3263, "bank": 0.472, "bank account": 0.3646, "batch confidential": 0.2563, "be": 0.0849, "benefits": -0.3117, "building": -0.2452, "candidates": 0.0285, "check": 0.3542, "classification": -0.3477, "client": 0.2934, "cloud": -0.4683, "company building": -0.2452, "computer": -0.3043, "confidential": 0.2563, "copy": 0.2934, "daily": 0.3893, "dashboards": -0.3117, "direct": 0.2492, "documentation": 0.0084, "earn rs": 0.2726, "enterprise": -0.2954, "equipment": 0.0571, "flask": -0.3099, "fund": -0.0222, "git": -0.2717, "guarantee": 0.2894, "home guaranteed": 0.283, "hybrid": -0.2634, "india": -0.0397, "interview required": 0.2492, "it": -0.0397, "junior": 0.0048, "laptop": 0.3386, "leading": 0.2759, "learn": -0.3477, "make": 0.283, "make week": 0.283, "mentorship": -0.3043, "mnc": 0.2492, "modern": -0.2789, "modern javascript": -0.2789, "monthly": -0.3016, "monthly stipend": -0.3016, "of web": -0.2789, "onboarding": 0.0285, "or interview": 0.2821, "part": 0.348, "part time": 0.348, "pay refundable": 0.2563, "pipelines": -0.3906, "portal": 0.0282, "processing": 0.2925, "product": -0.3169, "pune": -0.2954, "queries": -0.3479, "recruiter": 0.3646, "remote": 0.0571, "rest": -0.2254, "rest apis": -0.2254, "science": -0.2288, "scikit": -0.3477, "scikit learn": -0.3477, "security": 0.045, "selected": 0.3152, "send": 0.3646, "slot": 0.2492, "slot no": 0.2492, "slots": 0.3159, "tech": -0.2254, "telegram": 0.2993, "tests": -0.2699, "tools": -0.2452, "trainee": -0.0013, "training": -0.0013, "transfer": 0.3999, "typing": 0.2993, "understanding": -0.2818, "understanding of": -0.2818, "undisclosed": 0.3152, "urgent": 0.2726, "weekly": 0.3705, "whatsapp": 0.283, "within": 0.3152, "within hours": 0.3152, "write automated": -0.3263, "aadhaar": 0.1844, "aadhaar card": 0.1844, "ability": -0.1815, "ability to": -0.1815, "account login": 0.2494, "account to": 0.1844, "across": -0.213, "across major": -0.213, "act": 0.1915, "act immediately": 0.1915, "administrative": 0.2494, "administrative assistant": 0.2494, "adoption": -0.2031, "adoption metrics": -0.2031, "after first": 0.2114, "after registration": 0.1329, "agency": 0.1844, "agency personal": 0.1844, "ai": 0.2114, "ai data": 0.2114, "analyst": -0.2031, "analyst batch": -0.2031, "analytical": -0.2031, "analytical communication": -0.2031, "analytics": -0.162, "analytics consulting": -0.162}, "intercept": -0.2434};
  const STOP_SET = new Set(ML_MODEL.stop);

  const PORTAL_DISCLAIMER_REGEX =
    /(?:beware\s+of\s+imposters!(?:[^\n]*read\s+more|[^\n]*)|(?:\b(?:naukri\.com|internshala|indeed|linkedin|foundit)\b|\bfraudsters\s+may\s+ask\b)[^\n.]*?(?:registration\s+fee|refundable\s+fee|in\s+exchange\s+of\s+money)[^\n.]*\.?)/gi;

  const LOCAL_RULES = [
    {
      id: "RULE_UPFRONT_FEE",
      severity: "high",
      weight: 55,
      patterns: [
        /\b(?:registration|onboarding|application|processing|training|certification|exam|security|verification|background\s+check|documentation|id\s+card|joining|administrative)\s+(?:fee|fees|charge|charges|deposit|payment|cost)\b/gi,
        /\b(?:refundable|one[-\s]?time|nominal|mandatory)\s+(?:security\s+)?(?:deposit|fee|charge|payment)\b/gi,
        /\bpay\s+(?:a\s+)?(?:small\s+|nominal\s+|refundable\s+)?(?:fee|deposit|amount|charge)\b/gi,
        /\b(?:fee|deposit|payment)\s+of\s+(?:rs\.?|inr|\$|usd|£|€)\s*\d[\d,]*(?:\.\d+)?\b/gi,
        /(?:(?:\b(?:rs\.?|inr|usd)\b)|[\$£€])\s*\d[\d,]*(?:\.\d+)?\s+(?:for\s+|as\s+)?(?:registration|training|onboarding|security\s+deposit|processing|application\s+fee|laptop\s+deposit)\b/gi,
        /\b(?:pay\s+via\s+|send\s+via\s+)?(?:upi|gpay|google\s+pay|phonepe|paytm|zelle|venmo|cash\s*app)\s+(?:for\s+|to\s+pay\s+)?(?:fee|charges?|deposit|onboarding)\b/gi
      ],
      reason: "Asks you to pay money (registration, training, or security fee). Real companies pay you — they do not charge you to join.",
      action: "Never pay any fee or deposit for a job or internship, even if they call it refundable."
    },
    {
      id: "RULE_EQUIPMENT_OR_CHECK",
      severity: "high",
      weight: 50,
      patterns: [
        /\b(?:laptop|equipment|software|hardware|macbook|workstation|starter\s+kit)\s+(?:security\s+)?(?:deposit|fee|shipping\s+fee|advance\s+payment)\b/gi,
        /\b(?:we\s+will\s+send\s+you\s+a\s+check|deposit\s+the\s+check|purchase\s+equipment\s+from\s+our\s+(?:approved\s+)?vendor)\b/gi,
        /\b(?:wire\s+transfer|crypto(?:currency)?|bitcoin|usdt|gift\s+card|itunes\s+card|steam\s+card|western\s+union|zelle|venmo|cash\s*app|paytm|phonepe|gpay|google\s+pay)\s+(?:for\s+equipment|payment|to\s+complete\s+onboarding|to\s+our\s+vendor)\b/gi
      ],
      reason: "Asks for a laptop/equipment fee or tells you to deposit a check to buy supplies.",
      action: "Do not pay for company laptops or cash checks sent by strangers."
    },
    {
      id: "RULE_BANK_DETAILS",
      severity: "high",
      weight: 55,
      patterns: [
        /\b(?:bank\s+account\s+(?:login|password|credentials|details|number)|online\s+banking\s+(?:login|password|id)|credit\s+card\s+(?:number|details|cvv)|debit\s+card\s+(?:number|pin|cvv)|upi\s+pin|atm\s+pin|share\s+your\s+otp)\b/gi,
        /\b(?:provide|send|submit|share|email)\s+(?:your\s+)?(?:bank\s+account|bank\s+details|routing\s+number|credit\s+card|debit\s+card|bank\s+statement)\s+(?:before\s+interview|to\s+apply|with\s+your\s+application|immediately)\b/gi,
        /\b(?:use\s+your\s+personal\s+bank\s+account|receive\s+and\s+transfer\s+funds)\b/gi
      ],
      reason: "Asks for private banking details, card numbers, PIN/OTP, or wants to move money through your bank account.",
      action: "Never share bank logins, PINs, OTPs, or let anyone use your bank account."
    },
    {
      id: "RULE_ID_BEFORE_INTERVIEW",
      severity: "medium",
      weight: 35,
      patterns: [
        /\b(?:send|email|whatsapp|submit|upload|provide)\s+(?:a\s+copy\s+of\s+|scans?\s+of\s+|your\s+)?(?:social\s+security\s+number|\bssn\b|aadhaar\s+card|pan\s+card|passport\s+copy|driver.?s?\s+license)\s+(?:to\s+apply|before\s+interview|for\s+shortlisting|with\s+your\s+resume|immediately)\b/gi
      ],
      reason: "Asks for Aadhaar, PAN, passport, or ID scans before you even have an interview.",
      action: "Do not send ID card scans until you have interviewed and verified the company."
    },
    {
      id: "RULE_UNREALISTIC_PAY",
      severity: "high",
      weight: 45,
      patterns: [
        /\bearn\s+(?:up\s+to\s+|over\s+)?(?:rs\.?|inr|\$|usd|£|€)\s*\d[\d,]*\s*(?:per\s+day|daily|per\s+hour|per\s+week|weekly)\s+(?:working\s+from\s+home|part[-\s]?time|with\s+no\s+experience|working\s+\d+\s*[-–to]*\s*\d*\s*hours?)\b/gi,
        /\b(?:work\s+(?:only\s+|just\s+)?\d+\s*(?:to\s+\d+\s*)?hours?\s+(?:a|per)\s+day\s+and\s+earn|guaranteed\s+(?:daily|weekly)\s+(?:income|payout|earnings)|make\s+\$\d[\d,]+\s+(?:a\s+week|per\s+week|a\s+day|daily))\b/gi
      ],
      reason: "Promises unusually high daily/weekly pay for very little work or zero experience.",
      action: "Be careful with easy daily payout offers — check normal market pay for this role."
    },
    {
      id: "RULE_NO_INTERVIEW",
      severity: "medium",
      weight: 35,
      patterns: [
        /\b(?:100%\s*(?:job|placement|selection)\s*guarantee|guaranteed\s+(?:job\s+placement|internship\s+placement|offer\s+letter|hiring))\b/gi,
        /\b(?:no\s+interview\s+(?:required|needed)|direct\s+(?:joining|offer\s+letter|selection)\s+without\s+interview|instant\s+offer\s+letter|selected\s+without\s+(?:any\s+)?interview)\b/gi
      ],
      reason: "Promises a guaranteed job or instant offer letter without a real interview.",
      action: "Real employers interview you first. Question any offer letter sent without an interview."
    },
    {
      id: "RULE_URGENCY_PRESSURE",
      severity: "medium",
      weight: 25,
      patterns: [
        /\b(?:urgent\s+(?:requirement|hiring)\s*[-—:]?\s*(?:apply|join|confirm|respond)\s+(?:immediately|within\s+\d+\s*(?:hours?|minutes?)|today\s+only))\b/gi,
        /\b(?:confirm|accept|respond|pay|complete\s+registration)\s+within\s+(?:\d+|24|48)\s*(?:hours?|minutes?|hrs?)\s+(?:or\s+(?:your\s+)?(?:offer|seat|slot)\s+will\s+be\s+cancelled|to\s+secure\s+your\s+(?:seat|slot))\b/gi,
        /\b(?:only\s+\d+\s+(?:seats?|slots?|spots?)\s+left\s*[-—!,]?\s*(?:register|apply|pay|act)\s+(?:now|immediately|today)|act\s+immediately\s+before\s+slots?\s+fill)\b/gi
      ],
      reason: "Uses rush tactics (like confirm in 24 hours or lose your slot) to stop you from checking the company.",
      action: "Take your time to research the company — do not let countdown pressure rush you."
    },
    {
      id: "RULE_HIDDEN_COMPANY_OR_FREE_EMAIL",
      severity: "medium",
      weight: 30,
      patterns: [
        /\b[\w.+-]+@(?:gmail|yahoo|outlook|hotmail|aol|protonmail|rediffmail|ymail|icloud|mail)\.(?:com|in|co\.uk|net|org)\b/gi,
        /\b(?:company\s+name\s+(?:will\s+be\s+shared|disclosed)\s+after\s+(?:registration|payment|shortlisting)|undisclosed\s+employer)\b/gi,
        /\b(?:posted\s+by\s+freelancer|leading\s+management\s+consultant|india.?s\s+leading\s+mnc)\b/gi
      ],
      reason: "Hides the real company name (e.g. Posted by Freelancer or India's leading MNC) or uses a personal @gmail/@yahoo address.",
      action: "Ask for the exact company name and official website before sharing personal details."
    },
    {
      id: "RULE_WHATSAPP_TELEGRAM",
      severity: "medium",
      weight: 30,
      patterns: [
        /\b(?:contact|message|ping|reach\s+out|apply|interview|connect|send\s+your\s+resume|dm)\s+(?:us\s+|me\s+|hr\s+|the\s+hiring\s+manager\s+)?(?:only\s+)?(?:on|via|through)\s+(?:whatsapp|telegram|signal|wechat|viber)\b/gi,
        /\bt\.me\/[a-zA-Z0-9_]+\b/gi,
        /\b(?:whatsapp|telegram)\s+(?:only|number|hr|recruiter|interview|chat|your\s+resume|your\s+cv|your\s+details)\b/gi
      ],
      reason: "Asks you to interview or apply only through WhatsApp or Telegram instead of official company email.",
      action: "Ask the recruiter to email you from their official company domain."
    }
  ];

  function computeMlProbability(text) {
    const rawWords = (text.toLowerCase().match(/[a-z]{2,}/g) || []).filter(w => !STOP_SET.has(w));
    const tokens = [...rawWords];
    for (let i = 0; i < rawWords.length - 1; i++) {
      tokens.push(rawWords[i] + " " + rawWords[i + 1]);
    }

    const counts = {};
    for (const tok of tokens) {
      if (Object.prototype.hasOwnProperty.call(ML_MODEL.idf, tok)) {
        counts[tok] = (counts[tok] || 0) + 1;
      }
    }

    const tfidf = {};
    let normSq = 0;
    for (const tok of Object.keys(counts)) {
      const val = (1.0 + Math.log(counts[tok])) * ML_MODEL.idf[tok];
      tfidf[tok] = val;
      normSq += val * val;
    }

    const norm = Math.sqrt(normSq);
    if (norm === 0) {
      return 0;
    }

    let z = ML_MODEL.intercept;
    for (const tok of Object.keys(tfidf)) {
      z += (tfidf[tok] / norm) * ML_MODEL.coef[tok];
    }

    const prob = 1.0 / (1.0 + Math.exp(-Math.max(-30, Math.min(30, z))));
    return Math.round(prob * 100);
  }

  // DOM Elements
  const scanPageBtn = document.getElementById("scan-page-btn");
  const jobTextInput = document.getElementById("job-text-input");
  const analyzeBtn = document.getElementById("analyze-btn");
  const clearTextBtn = document.getElementById("clear-text-btn");
  const imageUploadInput = document.getElementById("image-upload-input");
  const analyzeImageBtn = document.getElementById("analyze-image-btn");

  const errorState = document.getElementById("error-state");
  const errorTitle = document.getElementById("error-title");
  const errorMessage = document.getElementById("error-message");

  const resultsView = document.getElementById("results-view");
  const riskHeadline = document.getElementById("risk-headline");
  const riskScorePill = document.getElementById("risk-score-pill");
  const riskOneLiner = document.getElementById("risk-one-liner");
  const mlScoreText = document.getElementById("ml-score-text");
  const flagsSection = document.getElementById("flags-section");
  const flagsList = document.getElementById("flags-list");
  const actionsList = document.getElementById("actions-list");

  function showError(title, msg) {
    errorTitle.textContent = title;
    errorMessage.textContent = msg;
    errorState.classList.remove("hidden");
  }

  function hideError() {
    errorState.classList.add("hidden");
  }

  function analyzeAndRender(rawText) {
    const trimmed = (rawText || "").trim();
    if (trimmed.length < 10) {
      showError("Text Too Short", "Open a job listing page or paste at least one sentence of the job offer.");
      return;
    }

    hideError();
    const cleaned = trimmed.replace(PORTAL_DISCLAIMER_REGEX, " ").trim();
    const words = cleaned.split(/\s+/).filter(Boolean).length;

    const triggered = [];
    let rawScore = 0;
    let hasHigh = false;

    for (const rule of LOCAL_RULES) {
      const matches = [];
      const seen = new Set();
      for (const pat of rule.patterns) {
        pat.lastIndex = 0;
        let m;
        while ((m = pat.exec(cleaned)) !== null) {
          const phrase = m[0].trim();
          if (phrase && !seen.has(phrase.toLowerCase())) {
            seen.add(phrase.toLowerCase());
            matches.push(phrase);
          }
        }
      }
      if (matches.length > 0) {
        rawScore += rule.weight;
        if (rule.severity === "high") hasHigh = true;
        triggered.push({ ...rule, matches });
      }
    }

    const score = Math.min(100, rawScore);
    const mlPct = computeMlProbability(cleaned);

    let level = "low";
    let headline = "No Obvious Red Flags";
    let oneLiner = "No common scam phrases found, but still check the company website before applying.";

    if (triggered.length === 0 && words < 20) {
      level = "low";
      headline = "Not Enough Text";
      oneLiner = "This snippet is very short. Paste the full job description for a better check.";
    } else if (hasHigh || score >= 50) {
      level = "high";
      headline = "High Scam Risk";
      oneLiner = `Found ${triggered.length} serious warning sign(s). Do not pay money or send private IDs.`;
    } else if (triggered.length > 0) {
      level = "medium";
      headline = "Caution — Check Carefully";
      oneLiner = `Found ${triggered.length} warning sign(s). Verify the company name and website before applying.`;
    }

    resultsView.classList.remove("hidden");
    riskHeadline.textContent = headline;
    riskHeadline.className = `risk-headline text-${level}`;

    riskScorePill.textContent = `Rule Risk: ${score}/100`;
    riskScorePill.className = `score-pill pill-${level}`;
    riskOneLiner.textContent = oneLiner;
    mlScoreText.textContent = `${mlPct}% similarity to scam patterns (estimate only, not proof)`;

    flagsList.innerHTML = "";
    if (triggered.length === 0) {
      flagsSection.classList.add("hidden");
    } else {
      flagsSection.classList.remove("hidden");
      triggered.forEach((item) => {
        const li = document.createElement("li");
        li.className = "flag-item";

        const reasonDiv = document.createElement("div");
        reasonDiv.className = "flag-reason";
        reasonDiv.textContent = item.reason;

        const wordsDiv = document.createElement("div");
        wordsDiv.className = "flag-words";
        wordsDiv.textContent = "Matched words: ";
        item.matches.slice(0, 3).forEach((w, idx) => {
          const code = document.createElement("code");
          code.textContent = `"${w}"`;
          wordsDiv.appendChild(code);
          if (idx < Math.min(item.matches.length, 3) - 1) {
            wordsDiv.appendChild(document.createTextNode(" "));
          }
        });

        li.appendChild(reasonDiv);
        li.appendChild(wordsDiv);
        flagsList.appendChild(li);
      });
    }

    actionsList.innerHTML = "";
    const actions = [];
    triggered.forEach((item) => {
      if (!actions.includes(item.action)) actions.push(item.action);
    });
    const defaults = [
      "Check that the job is listed on the company's official website.",
      "Make sure recruiter emails come from an official company domain (not @gmail.com).",
      "Never pay any money for registration, training, or laptops."
    ];
    for (const d of defaults) {
      if (actions.length < 3 && !actions.includes(d)) actions.push(d);
    }
    actions.slice(0, 3).forEach((act) => {
      const li = document.createElement("li");
      li.textContent = act;
      actionsList.appendChild(li);
    });
  }

  async function scanAndCheckCurrentPage(isAuto = false) {
    hideError();
    if (typeof chrome === "undefined" || !chrome.tabs || !chrome.scripting) {
      if (!isAuto) {
        showError("Paste Text Above", "Tab scanning is only available inside the Chrome extension.");
      }
      return;
    }

    try {
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      const activeTab = tabs && tabs[0];
      const tabUrl = (activeTab && activeTab.url) || "";

      if (
        !activeTab ||
        !activeTab.id ||
        tabUrl.startsWith("chrome://") ||
        tabUrl.startsWith("edge://") ||
        tabUrl.startsWith("chrome-extension://") ||
        tabUrl.includes("chromewebstore.google.com")
      ) {
        if (!isAuto) {
          showError(
            "Switch to a Job Tab",
            "Chrome blocks scanning on chrome:// settings pages. Switch to your job listing tab (like Naukri or LinkedIn) or paste text above."
          );
        }
        return;
      }

      const injection = await chrome.scripting.executeScript({
        target: { tabId: activeTab.id },
        func: () => {
          const selectors = [
            "#jobDescriptionText",
            ".jobs-description__content",
            ".jobs-box__html-content",
            ".jobsearch-JobComponent-description",
            ".job-desc-container",
            ".dang-inner-html",
            ".internship_details",
            ".job_details",
            ".job-description",
            "#job-description",
            "article",
            "main",
            "[role='main']",
            "body"
          ];
          for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el && el.innerText && el.innerText.trim().length >= 60) {
              return el.innerText.trim().slice(0, 24000);
            }
          }
          return document.body ? document.body.innerText.trim().slice(0, 24000) : "";
        }
      });

      const extracted = injection && injection[0] && injection[0].result;
      if (!extracted || extracted.length < 10) {
        if (!isAuto) {
          showError("No Job Text Found", "Could not read text on this page. Paste the job description in the box above.");
        }
        return;
      }

      jobTextInput.value = extracted;
      analyzeAndRender(extracted);
    } catch (_err) {
      if (!isAuto) {
        showError("Cannot Read This Page", "Copy and paste the job text into the box above.");
      }
    }
  }

  async function analyzeScreenshot() {
    const file = imageUploadInput.files && imageUploadInput.files[0];
    if (!file) {
      showError("No Screenshot Selected", "Choose a PNG or JPG screenshot file first.");
      return;
    }
    hideError();
    try {
      const formData = new FormData();
      formData.append("image", file);
      const resp = await fetch(`${API_BASE_URL}/api/analyze-image`, {
        method: "POST",
        body: formData
      });
      const data = await resp.json().catch(() => null);
      if (!resp.ok || !data) {
        showError("Screenshot Check Failed", (data && data.message) || "Could not read text from screenshot.");
        return;
      }
      const extracted = (data.ocr_extraction && data.ocr_extraction.extracted_text) || "";
      jobTextInput.value = extracted;
      analyzeAndRender(extracted);
    } catch (_err) {
      showError(
        "Start Python Server for Screenshots",
        "Page scanning and text checks run directly in Chrome, but screenshot OCR needs 'python app.py' running."
      );
    }
  }

  scanPageBtn.addEventListener("click", () => scanAndCheckCurrentPage(false));
  analyzeBtn.addEventListener("click", () => analyzeAndRender(jobTextInput.value));
  clearTextBtn.addEventListener("click", () => {
    jobTextInput.value = "";
    resultsView.classList.add("hidden");
    hideError();
  });
  analyzeImageBtn.addEventListener("click", analyzeScreenshot);

  scanAndCheckCurrentPage(true);
})();
