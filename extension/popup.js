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
  const ML_MODEL = {"stop": ["a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"], "idf": {"quality": 3.783, "scheme": 4.4761, "commercial": 3.783, "retail": 3.6288, "agency": 3.0098, "receive": 2.7415, "shopping": 4.4761, "assignments": 4.1884, "mail": 4.4761, "send": 2.8667, "check": 3.1768, "800": 3.2721, "deposit": 2.7415, "bank": 2.4392, "500": 2.8667, "commission": 3.1768, "purchase": 3.3775, "apple": 4.4761, "gift": 4.1884, "cards": 4.4761, "local": 3.9653, "stores": 4.4761, "photos": 4.4761, "card": 2.8021, "codes": 4.1884, "telegram": 3.2721, "recruiter": 3.783, "account": 2.1407, "company": 2.7415, "checks": 4.4761, "smartphone": 3.4953, "camera": 4.4761, "assignment": 4.1884, "flexible": 3.1768, "time": 2.4837, "hours": 2.8021, "send check": 4.4761, "check 800": 4.4761, "800 deposit": 4.4761, "deposit check": 3.6288, "check bank": 4.4761, "send photos": 4.4761, "gift card": 4.4761, "bank account": 2.5302, "flexible time": 4.1884, "product": 3.9653, "marketing": 3.6288, "associate": 3.0898, "management": 3.2721, "software": 3.0098, "vendor": 3.1768, "creative": 3.783, "design": 3.4953, "conduct": 3.4953, "feature": 4.4761, "create": 4.4761, "sales": 3.783, "coordinate": 2.9357, "release": 4.4761, "interview": 2.3558, "video": 3.783, "case": 4.1884, "studies": 4.1884, "bachelor": 2.6843, "degree": 2.3967, "communications": 3.9653, "strong": 3.1768, "skills": 2.7415, "analytical": 4.4761, "competitive": 2.9357, "salary": 2.8021, "annual": 3.4953, "professional": 4.4761, "development": 4.4761, "stipend": 3.6288, "health": 2.2789, "dental": 3.2721, "management software": 4.4761, "software vendor": 3.9653, "creative design": 4.4761, "bachelor degree": 3.2721, "communications strong": 4.4761, "competitive salary": 3.6288, "salary annual": 4.4761, "health dental": 3.783, "typing": 3.783, "copy": 3.783, "paste": 4.4761, "executive": 3.6288, "leading": 4.1884, "multinational": 4.4761, "client": 3.0898, "hiring": 3.0098, "immediately": 3.0898, "work": 2.6303, "daily": 2.5302, "contact": 3.6288, "hr": 3.0898, "whatsapp": 3.2721, "instant": 3.1768, "offer": 3.2721, "letter": 3.4953, "zero": 3.9653, "experience": 1.8611, "needed": 3.0898, "required": 3.0898, "make": 3.0898, "000": 2.6303, "week": 2.9357, "easily": 3.6288, "home": 1.9111, "guaranteed": 2.3558, "payout": 3.0898, "copy paste": 4.4761, "work hours": 3.9653, "contact hr": 4.4761, "hr whatsapp": 4.1884, "instant offer": 4.1884, "offer letter": 3.4953, "zero experience": 3.9653, "experience needed": 3.4953, "make 000": 3.783, "000 week": 3.9653, "easily home": 4.1884, "home guaranteed": 3.0898, "guaranteed daily": 3.6288, "daily payout": 3.783, "junior": 3.6288, "data": 3.3775, "engineer": 3.783, "streaming": 4.4761, "media": 3.4953, "platform": 3.6288, "delivering": 4.4761, "audio": 4.1884, "content": 4.1884, "build": 3.6288, "optimize": 3.9653, "batch": 4.4761, "real": 4.1884, "using": 2.9357, "structure": 4.4761, "computer": 3.6288, "science": 3.3775, "python": 3.9653, "sql": 4.1884, "understanding": 3.9653, "concepts": 4.1884, "free": 3.4953, "premium": 4.4761, "medical": 3.0898, "coverage": 3.0098, "internet": 4.1884, "subsidy": 4.1884, "audio video": 4.4761, "real time": 4.1884, "computer science": 4.4761, "salary medical": 4.4761, "medical coverage": 4.1884, "human": 4.4761, "resources": 4.4761, "assistant": 2.8021, "immediate": 3.0098, "selection": 3.9653, "corporate": 3.4953, "consultant": 4.1884, "undisclosed": 3.9653, "selected": 3.4953, "candidates": 3.9653, "email": 3.6288, "scans": 4.4761, "social": 3.3775, "security": 3.3775, "driver": 3.6288, "license": 3.4953, "passport": 4.1884, "desk": 3.783, "com": 3.9653, "scheduling": 4.1884, "phone": 2.9357, "provide": 3.4953, "id": 3.783, "government": 3.4953, "initial": 3.783, "resume": 3.9653, "high": 2.7415, "weekly": 2.5302, "200": 3.6288, "100": 3.2721, "remote": 2.6843, "job": 3.9653, "human resources": 4.4761, "consultant undisclosed": 4.4761, "selected candidates": 3.9653, "social security": 3.9653, "driver license": 3.783, "phone interview": 4.4761, "interview immediate": 4.4761, "id scans": 4.4761, "high weekly": 4.1884, "weekly salary": 3.9653, "200 week": 4.4761, "100 remote": 4.4761, "rating": 4.4761, "review": 3.783, "specialist": 3.1768, "cash": 3.783, "film": 4.4761, "distribution": 3.9653, "analytics": 4.4761, "group": 3.4953, "rate": 4.1884, "released": 4.4761, "online": 2.8021, "paid": 2.8667, "complete": 3.6288, "20": 4.4761, "tasks": 4.4761, "day": 3.2721, "payouts": 4.4761, "membership": 4.1884, "fee": 2.8021, "wire": 3.0098, "transfer": 2.6843, "unlock": 4.1884, "knowledge": 3.3775, "earn": 2.0782, "900": 4.4761, "working": 2.7415, "45": 3.783, "minutes": 4.4761, "rating tasks": 4.4761, "high daily": 4.1884, "200 wire": 4.4761, "wire transfer": 3.2721, "phone earn": 4.4761, "day working": 3.9653, "entry": 3.9653, "joining": 3.2721, "confidential": 4.1884, "registration": 3.783, "urgent": 3.9653, "direct": 3.3775, "type": 4.4761, "documents": 3.783, "pay": 2.6303, "refundable": 3.1768, "120": 4.4761, "secure": 4.4761, "slot": 4.4761, "accepted": 4.4761, "hour": 3.6288, "placement": 3.4953, "guarantee": 4.1884, "data entry": 4.4761, "immediate joining": 3.9653, "direct joining": 3.9653, "documents home": 4.1884, "home selected": 4.4761, "candidates pay": 4.4761, "pay refundable": 3.3775, "refundable registration": 4.1884, "fee 120": 4.4761, "secure slot": 4.4761, "earn 800": 4.1884, "daily working": 3.9653, "working hour": 4.4761, "hour home": 4.1884, "payout 100": 4.4761, "100 placement": 4.4761, "placement guarantee": 4.4761, "tutoring": 4.4761, "global": 3.1768, "education": 3.783, "private": 3.1768, "assist": 2.8021, "students": 4.4761, "basic": 3.3775, "hourly": 3.4953, "50": 3.9653, "rs": 3.6288, "activate": 3.9653, "profile": 4.4761, "teaching": 4.4761, "degree required": 4.4761, "hourly rate": 4.4761, "registration deposit": 4.4761, "deposit rs": 4.4761, "required earn": 4.1884, "earn rs": 3.6288, "rs 000": 4.1884, "000 daily": 3.9653, "daily home": 4.4761, "home instant": 4.1884, "program": 3.9653, "coordinator": 3.1768, "community": 3.9653, "center": 3.783, "school": 3.2721, "sports": 4.4761, "plan": 3.2721, "supervise": 4.4761, "activities": 4.4761, "volunteer": 4.1884, "academic": 3.9653, "organize": 3.9653, "maintain": 3.3775, "student": 3.9653, "safety": 4.1884, "records": 4.4761, "recreation": 4.4761, "groups": 4.4761, "benefits": 2.579, "family": 3.9653, "gym": 4.4761, "non": 4.4761, "team": 3.6288, "environment": 3.9653, "coordinate volunteer": 4.4761, "maintain student": 4.4761, "health benefits": 3.0898, "benefits free": 4.4761, "gym membership": 4.4761, "paid time": 3.9653, "document": 3.6288, "reviewer": 3.9653, "partner": 3.783, "proofread": 4.4761, "simple": 4.4761, "english": 3.6288, "training": 3.3775, "onboarding": 3.783, "150": 4.4761, "zelle": 3.6288, "venmo": 3.9653, "receiving": 4.1884, "files": 3.6288, "today": 3.3775, "formal": 3.783, "background": 3.9653, "personal": 2.579, "laptop": 4.4761, "600": 3.9653, "document reviewer": 4.4761, "private client": 4.4761, "simple english": 4.4761, "refundable training": 4.4761, "onboarding fee": 4.4761, "fee 150": 4.4761, "150 zelle": 4.4761, "zelle venmo": 4.4761, "direct offer": 4.4761, "earn 600": 4.4761, "600 daily": 4.4761, "working home": 3.783, "guaranteed weekly": 4.4761, "weekly payout": 4.1884, "logistics": 3.9653, "freight": 4.1884, "party": 4.4761, "book": 4.4761, "schedule": 3.6288, "negotiate": 4.4761, "market": 4.1884, "rates": 4.4761, "track": 4.4761, "transit": 3.9653, "resolve": 4.1884, "years": 3.0898, "communication": 4.1884, "base": 3.783, "plus": 3.6288, "performance": 4.1884, "401": 2.8667, "communication skills": 4.4761, "competitive base": 4.1884, "base salary": 4.1884, "salary plus": 4.4761, "benefits 401": 4.4761, "401 plan": 4.1884, "bilingual": 4.1884, "translation": 4.4761, "language": 4.4761, "translate": 4.1884, "short": 4.4761, "apply": 3.783, "scan": 3.9653, "banking": 3.3775, "documentation": 3.6288, "upfront": 4.4761, "home apply": 4.4761, "hr flexible": 4.4761, "flexible schedule": 4.4761, "cryptocurrency": 3.9653, "investment": 3.9653, "research": 3.783, "trainee": 3.6288, "digital": 3.1768, "asset": 4.4761, "arbitrage": 4.4761, "trading": 4.1884, "use": 3.9653, "accounts": 4.1884, "process": 3.1768, "deposits": 3.9653, "credentials": 3.9653, "channel": 4.4761, "qualify": 4.4761, "crypto": 3.783, "prior": 3.9653, "finance": 3.6288, "trainee digital": 4.4761, "digital asset": 4.4761, "asset arbitrage": 4.4761, "use personal": 3.9653, "personal bank": 2.8667, "government id": 4.4761, "online banking": 3.6288, "earn 500": 3.4953, "500 weekly": 3.9653, "weekly commission": 4.4761, "compliance": 4.1884, "analyst": 4.4761, "level": 4.4761, "firm": 3.783, "managing": 4.4761, "equity": 4.1884, "funds": 3.2721, "employee": 3.2721, "verify": 3.783, "fund": 4.1884, "quarterly": 4.1884, "economics": 4.4761, "attention": 4.1884, "generous": 3.6288, "bonus": 3.4953, "tuition": 3.783, "assistance": 4.1884, "entry level": 4.4761, "tuition assistance": 4.4761, "administrative": 3.4953, "business": 3.6288, "service": 3.2721, "department": 4.4761, "cashier": 4.4761, "office": 3.4953, "equipment": 3.3775, "macbook": 4.4761, "pro": 4.4761, "ready": 4.4761, "cashier check": 4.4761, "check purchase": 4.4761, "home office": 4.1884, "macbook pro": 4.4761, "vendor deposit": 4.1884, "account ready": 4.4761, "salary 500": 4.1884, "500 week": 4.1884, "flexible hours": 4.1884, "hours home": 3.9653, "purchasing": 4.4761, "setup": 3.9653, "advance": 3.783, "remaining": 4.4761, "accredited": 4.4761, "technology": 3.783, "access": 4.4761, "atm": 4.4761, "35": 4.1884, "zelle wire": 4.4761, "deposit earn": 4.4761, "payment": 3.0898, "processing": 3.6288, "consumer": 4.1884, "payments": 4.1884, "checking": 4.4761, "applicants": 3.9653, "submit": 3.9653, "login": 3.6288, "details": 3.6288, "debit": 3.9653, "number": 3.6288, "cvv": 4.1884, "pin": 4.4761, "350": 4.4761, "payment processing": 4.4761, "personal checking": 4.4761, "checking account": 4.4761, "selected applicants": 4.4761, "applicants submit": 4.4761, "bank login": 4.4761, "login details": 4.4761, "debit card": 3.9653, "formal interview": 4.1884, "interview required": 4.4761, "form": 4.4761, "portal": 3.6288, "left": 4.1884, "act": 4.4761, "slots": 4.1884, "exam": 4.1884, "project": 3.9653, "reach": 4.4761, "confirm": 4.1884, "seat": 4.4761, "urgent hiring": 4.1884, "left act": 4.4761, "act immediately": 4.4761, "interview pay": 4.1884, "pay time": 4.4761, "time onboarding": 4.1884, "receive project": 4.4761, "project files": 4.4761, "reach hr": 4.4761, "basic typing": 4.4761, "week guaranteed": 4.4761, "security deposit": 4.1884, "legal": 4.4761, "contracts": 4.1884, "role": 4.1884, "aadhaar": 3.9653, "pan": 3.9653, "statement": 4.4761, "receipt": 4.4761, "application": 4.4761, "monthly": 3.783, "aadhaar card": 3.9653, "card pan": 4.1884, "pan card": 3.9653, "card bank": 4.4761, "placement guaranteed": 4.4761, "inventory": 3.9653, "control": 4.4761, "parts": 4.4761, "distributor": 4.1884, "80": 4.4761, "perform": 3.9653, "reconcile": 4.1884, "stock": 4.1884, "clean": 3.783, "warehouse": 4.4761, "diploma": 3.9653, "operating": 4.1884, "barcode": 4.4761, "wage": 4.4761, "overtime": 3.9653, "discount": 4.1884, "high school": 3.783, "school diploma": 3.9653, "hourly wage": 4.4761, "medical dental": 4.1884, "database": 4.4761, "state": 3.6288, "information": 4.1884, "monitor": 3.9653, "oracle": 4.4761, "postgresql": 4.4761, "execute": 3.9653, "backup": 4.4761, "verification": 4.1884, "tests": 4.4761, "manage": 3.4953, "user": 4.4761, "tax": 4.4761, "apps": 4.4761, "cs": 4.4761, "linux": 4.4761, "administration": 4.4761, "pension": 3.6288, "sick": 4.4761, "days": 3.9653, "vacation": 4.4761, "healthcare": 3.3775, "degree cs": 4.4761, "financial": 3.9653, "transaction": 4.4761, "international": 4.1884, "transfers": 4.4761, "bitcoin": 3.783, "usdt": 3.9653, "western": 4.1884, "union": 3.9653, "partners": 4.4761, "routing": 3.783, "credit": 3.9653, "finalize": 4.1884, "employment": 3.9653, "active": 3.6288, "availability": 3.9653, "10": 4.1884, "processed": 4.4761, "associate international": 4.4761, "work home": 3.783, "client funds": 4.4761, "account receive": 3.9653, "transfer bitcoin": 4.4761, "western union": 4.1884, "banking login": 4.4761, "routing number": 4.1884, "credit card": 3.9653, "finalize employment": 4.1884, "active personal": 4.1884, "account online": 4.4761, "immediate availability": 3.9653, "earn 10": 4.4761, "10 commission": 4.4761, "000 weekly": 3.783, "kit": 4.4761, "85": 4.4761, "cashapp": 4.4761, "paypal": 4.4761, "friends": 4.4761, "ability": 3.783, "instant selection": 4.4761, "fee 85": 4.4761, "paypal friends": 4.4761, "friends family": 4.4761, "typing ability": 4.4761, "availability make": 4.4761, "make 800": 3.9653, "800 week": 4.4761, "week home": 4.4761, "100 job": 4.4761, "customer": 3.4953, "chat": 3.783, "representative": 3.783, "commerce": 4.4761, "support": 3.783, "text": 3.4953, "manager": 3.6288, "begin": 4.4761, "willing": 4.4761, "share": 3.9653, "necessary": 4.4761, "customer service": 3.9653, "customer support": 4.4761, "text interview": 4.1884, "hiring manager": 4.1884, "send copy": 4.4761, "background check": 4.4761, "check fee": 4.4761, "smartphone telegram": 4.4761, "weekly working": 4.1884, "working time": 4.1884, "time home": 3.783, "home experience": 3.783, "experience necessary": 4.4761, "virtual": 4.1884, "agent": 3.9653, "contract": 4.4761, "app": 3.9653, "send resume": 4.4761, "telegram app": 4.4761, "home zero": 4.4761, "broadcast": 4.4761, "production": 4.1884, "regional": 3.6288, "public": 3.3775, "operate": 4.4761, "studio": 3.9653, "live": 4.1884, "podcast": 4.4761, "journalism": 4.4761, "employee benefits": 4.1884, "automated": 3.9653, "start": 4.4761, "date": 4.4761, "willingness": 3.9653, "bonuses": 4.1884, "home private": 4.4761, "private investment": 4.4761, "login credentials": 4.1884, "start date": 4.4761, "account willingness": 4.1884, "willingness share": 4.4761, "store": 4.1884, "specialty": 4.4761, "outdoor": 4.1884, "goods": 4.4761, "specializing": 4.1884, "train": 4.1884, "associates": 4.4761, "technical": 3.783, "gear": 4.1884, "handling": 4.1884, "options": 4.4761, "service experience": 4.4761, "mid": 4.4761, "property": 4.1884, "patent": 4.4761, "draft": 3.9653, "discovery": 4.4761, "deadlines": 4.4761, "record": 4.1884, "comprehensive": 3.2721, "pass": 4.4761, "transit pass": 4.4761, "cloud": 3.9653, "enterprise": 4.4761, "saas": 4.4761, "applications": 3.9653, "pipeline": 4.4761, "targeted": 4.4761, "outreach": 4.4761, "linkedin": 4.1884, "energy": 4.4761, "excellent": 4.1884, "crm": 4.4761, "familiarity": 3.783, "uncapped": 4.4761, "career": 4.4761, "account executive": 4.4761, "care": 4.1884, "remote customer": 4.4761, "interview whatsapp": 4.4761, "account login": 4.4761, "manager whatsapp": 4.4761, "whatsapp account": 4.1884, "weekly home": 4.1884, "map": 4.4761, "tools": 4.1884, "surveys": 4.4761, "biology": 4.1884, "insurance": 3.2721, "field": 3.4953, "travel": 3.6288, "health insurance": 3.6288, "videos": 4.4761, "mobile": 3.1768, "withdraw": 4.4761, "earnings": 4.4761, "unlimited": 4.1884, "earning": 4.4761, "potential": 4.1884, "digital media": 4.4761, "agency earn": 4.4761, "500 000": 4.4761, "mobile phone": 3.6288, "daily earnings": 4.4761, "deposit administrative": 4.4761, "administrative fee": 4.1884, "usdt bitcoin": 4.4761, "telegram account": 4.4761, "unlimited earning": 4.4761, "earning potential": 4.4761, "task": 4.1884, "scam": 4.4761, "income": 4.1884, "ratings": 4.4761, "higher": 4.4761, "recharge": 4.4761, "wallet": 4.1884, "readiness": 4.4761, "mobile app": 4.4761, "daily income": 4.1884, "500 day": 4.4761, "general": 4.4761, "building": 4.1884, "educational": 4.1884, "facilities": 4.4761, "site": 4.4761, "engineering": 3.6288, "architecture": 4.4761, "read": 4.4761, "drawings": 4.4761, "vehicle": 4.4761, "allowance": 4.1884, "match": 4.1884, "ability read": 4.4761, "company vehicle": 4.4761, "insurance 401": 4.4761, "401 match": 4.4761, "buy": 4.1884, "exchange": 4.1884, "security number": 4.1884, "number bank": 4.1884, "president": 4.4761, "recruitment": 3.9653, "supplier": 4.1884, "executive assistant": 4.4761, "personal assistant": 4.4761, "receive advance": 4.4761, "active bank": 4.4761, "assistants": 4.4761, "ssn": 4.4761, "gmail": 4.4761, "payroll": 4.4761, "previous": 4.1884, "account number": 4.4761, "gmail com": 4.4761, "fast": 4.1884, "overseas": 4.4761, "password": 4.4761, "transactions": 4.4761, "receive client": 4.4761, "issue": 4.4761, "certified": 4.1884, "specialized": 4.4761, "laboratory": 3.9653, "analysis": 4.4761, "approved": 4.4761, "hiring remote": 4.4761, "payment vendor": 4.4761, "home flexible": 4.4761, "evening": 4.4761, "shift": 3.9653, "95": 4.4761, "guaranteed payout": 4.4761, "physical": 4.4761, "clinic": 4.4761, "prepare": 3.9653, "patient": 3.9653, "sanitize": 4.4761, "supplies": 4.4761, "greet": 4.4761, "pursuing": 4.4761, "certification": 4.1884, "compensation": 4.4761, "pursuing degree": 4.4761, "benefits time": 4.4761, "developing": 4.1884, "platforms": 4.4761, "candidate": 4.4761, "source": 4.4761, "life": 4.1884, "sciences": 4.4761, "terminology": 4.4761, "company developing": 4.1884, "experience understanding": 4.4761, "life sciences": 4.4761, "stock purchase": 4.4761, "purchase plan": 4.4761, "hotel": 4.1884, "write": 4.1884, "reviews": 4.4761, "luxury": 3.9653, "worldwide": 4.4761, "balance": 4.1884, "develop": 4.1884, "lead": 4.4761, "interactive": 4.4761, "gallery": 4.4761, "tours": 4.4761, "discounts": 4.4761, "science education": 4.4761, "healthcare coverage": 4.1884, "clinical": 4.4761, "conducting": 4.4761, "forms": 4.4761, "university": 4.4761, "pto": 4.4761, "public health": 4.4761, "benefits tuition": 4.4761, "days pto": 4.4761, "test": 3.6288, "web": 4.4761, "reports": 4.4761, "help": 4.1884, "suites": 4.4761, "testing": 3.783, "api": 4.4761, "inspection": 4.4761, "matching": 3.6288, "retirement": 4.4761, "dental coverage": 4.1884, "supply": 4.4761, "wholesale": 4.4761, "accounting": 4.4761, "valid": 4.1884, "orders": 4.1884, "wholesale distributor": 4.4761, "week time": 4.4761, "exclusively": 4.4761, "exclusively whatsapp": 4.4761, "working hours": 4.4761, "charge": 4.4761, "joining today": 3.783, "onboarding charge": 4.4761, "hold": 4.4761, "12": 4.4761, "provide bank": 4.4761, "card details": 4.4761, "funds earn": 4.4761, "residential": 4.4761, "prospective": 4.4761, "friendly": 3.9653, "discounted": 4.1884, "management company": 4.4761, "providing": 3.783, "centers": 4.4761, "fluent": 4.4761, "leave": 4.4761, "bilingual english": 4.4761, "valid driver": 4.4761, "pension comprehensive": 4.4761, "comprehensive health": 3.783, "benefits generous": 4.4761, "guest": 4.4761, "moneygram": 4.4761, "union moneygram": 4.4761, "interview needed": 4.4761, "developer": 4.1884, "cross": 4.1884, "ios": 4.4761, "implement": 4.1884, "micro": 4.1884, "integrate": 4.4761, "google": 3.9653, "apis": 4.4761, "provided": 3.9653, "years experience": 4.4761, "knowledge competitive": 4.4761, "equipment provided": 4.1884, "employer": 4.4761, "screening": 4.4761, "social media": 4.1884, "payout global": 4.4761, "global marketing": 4.4761, "google pay": 4.4761, "slots left": 4.4761, "needed earn": 4.4761, "boutique": 4.4761, "consultancy": 3.9653, "print": 4.4761, "metrics": 3.9653, "writing": 4.1884, "retreat": 4.4761, "annual bonus": 4.4761, "clerk": 4.4761, "110": 4.4761, "800 weekly": 4.1884, "network": 4.1884, "text chat": 4.4761, "whatsapp text": 4.4761, "750": 4.4761, "submission": 4.4761, "earn 750": 4.4761, "750 day": 4.4761, "delivery": 4.4761, "click": 4.4761, "order": 4.4761, "css": 4.4761, "responsive": 4.4761, "hybrid": 3.9653, "policy": 4.4761, "401 matching": 3.783, "technician": 3.9653, "samples": 4.1884, "chain": 4.4761, "chemistry": 4.4761, "electrical": 4.4761, "hardware": 3.9653, "assemble": 4.4761, "assemblies": 4.4761, "lab": 4.4761, "electrical engineering": 4.4761, "power": 4.4761, "arm": 4.4761, "bs": 4.4761, "proficient": 4.1884, "budget": 4.4761, "bs computer": 4.4761, "learning": 4.4761, "district": 4.1884, "school district": 4.4761, "foreign": 4.4761, "clients": 4.1884, "personal account": 4.4761, "account transfer": 4.4761, "interview active": 4.4761, "graphic": 4.4761, "portfolio": 4.4761, "graphic design": 4.4761, "assistant telegram": 4.4761, "software license": 4.4761, "license fee": 4.4761, "software fee": 4.4761, "month": 4.4761, "hours day": 4.4761, "day earn": 4.4761, "weekly easily": 4.1884, "agents": 4.4761, "earn 000": 4.4761, "ad": 4.4761, "intern": 4.1884, "structured": 4.4761, "senior": 4.1884, "decks": 4.4761, "excel": 4.4761, "mo": 4.4761, "mentorship": 4.4761, "assist senior": 4.4761, "monthly stipend": 4.4761, "wellness": 4.4761, "agricultural": 4.4761, "employing": 4.4761, "open": 4.4761, "clinics": 4.4761, "coverage 401": 4.4761, "solutions": 4.4761, "trip": 4.4761, "degree years": 4.4761, "code": 4.4761, "services": 4.1884, "operations": 3.9653, "recovery": 4.4761, "git": 4.4761, "brand": 4.4761, "collaborate": 4.4761, "figma": 4.4761, "hybrid schedule": 4.4761, "related": 4.4761, "25": 4.4761, "related field": 4.4761, "comprehensive medical": 4.4761, "holiday": 4.4761, "generous holiday": 4.4761, "help process": 4.4761, "trade": 4.4761, "manufacturing": 4.4761, "components": 4.4761, "experience familiarity": 4.4761, "pos": 4.4761, "issues": 4.4761, "differential": 4.4761, "shift differential": 4.4761, "health coverage": 4.4761, "netsuite": 4.4761, "eligibility": 4.1884, "competitive hourly": 4.4761, "staff": 4.4761, "reimbursement": 4.4761, "reimbursement 401": 4.4761, "office operations": 4.4761, "advanced": 4.4761, "producing": 4.4761}, "coef": {"quality": -0.0114, "scheme": 0.0071, "commercial": -0.0088, "retail": -0.017, "agency": 0.0088, "receive": 0.037, "shopping": 0.0077, "assignments": 0.0034, "mail": 0.0081, "send": 0.0355, "check": 0.0437, "800": 0.0272, "deposit": 0.0454, "bank": 0.0642, "500": 0.0279, "commission": 0.0232, "purchase": 0.0055, "apple": -0.0002, "gift": 0.0177, "cards": 0.0116, "local": 0.0071, "stores": -0.0007, "photos": 0.0075, "card": 0.0325, "codes": 0.011, "telegram": 0.0411, "recruiter": 0.0166, "account": 0.0683, "company": -0.0207, "checks": 0.0112, "smartphone": 0.0227, "camera": -0.0007, "assignment": 0.0108, "flexible": 0.0166, "time": -0.0015, "hours": 0.021, "send check": 0.0077, "check 800": 0.0075, "800 deposit": 0.0075, "deposit check": 0.0183, "check bank": 0.0071, "send photos": 0.0075, "gift card": 0.0074, "bank account": 0.0558, "flexible time": 0.0031, "product": -0.0101, "marketing": 0.0001, "associate": -0.0114, "management": -0.0156, "software": -0.0018, "vendor": 0.0048, "creative": -0.0163, "design": -0.0207, "conduct": -0.0166, "feature": -0.0088, "create": -0.0091, "sales": -0.0228, "coordinate": -0.0379, "release": -0.0003, "interview": 0.0466, "video": -0.0042, "case": -0.0114, "studies": -0.0119, "bachelor": -0.0429, "degree": -0.0399, "communications": -0.0172, "strong": -0.0299, "skills": -0.0278, "analytical": -0.0089, "competitive": -0.0321, "salary": -0.0147, "annual": -0.0251, "professional": -0.0083, "development": -0.0084, "stipend": -0.0216, "health": -0.0636, "dental": -0.0281, "management software": -0.0082, "software vendor": -0.0074, "creative design": -0.0004, "bachelor degree": -0.0277, "communications strong": -0.0083, "competitive salary": -0.0208, "salary annual": -0.0084, "health dental": -0.0173, "typing": 0.0214, "copy": 0.0154, "paste": 0.0075, "executive": 0.0047, "leading": 0.0106, "multinational": 0.0079, "client": 0.018, "hiring": 0.0279, "immediately": 0.0278, "work": 0.0194, "daily": 0.0384, "contact": 0.0118, "hr": 0.0233, "whatsapp": 0.042, "instant": 0.0284, "offer": 0.0194, "letter": 0.0209, "zero": 0.0147, "experience": -0.0107, "needed": 0.0294, "required": 0.0321, "make": 0.0291, "000": 0.0332, "week": 0.0279, "easily": 0.0188, "home": 0.0741, "guaranteed": 0.0497, "payout": 0.0305, "copy paste": 0.0075, "work hours": 0.014, "contact hr": 0.0074, "hr whatsapp": 0.01, "instant offer": 0.0114, "offer letter": 0.0209, "zero experience": 0.0147, "experience needed": 0.0216, "make 000": 0.0159, "000 week": 0.0138, "easily home": 0.0105, "home guaranteed": 0.029, "guaranteed daily": 0.0183, "daily payout": 0.0155, "junior": -0.0068, "data": -0.012, "engineer": -0.0183, "streaming": -0.0117, "media": -0.0175, "platform": -0.0143, "delivering": -0.0081, "audio": -0.0036, "content": -0.0044, "build": -0.0206, "optimize": -0.0146, "batch": 0.0005, "real": -0.0117, "using": -0.0317, "structure": -0.0081, "computer": -0.0007, "science": -0.0351, "python": -0.0227, "sql": -0.014, "understanding": -0.0149, "concepts": -0.0119, "free": -0.0225, "premium": -0.0088, "medical": -0.0284, "coverage": -0.0334, "internet": 0.0029, "subsidy": -0.0118, "audio video": -0.0085, "real time": -0.0117, "computer science": -0.0079, "salary medical": -0.0081, "medical coverage": -0.0112, "human": -0.0004, "resources": -0.0004, "assistant": 0.021, "immediate": 0.0366, "selection": 0.0152, "corporate": 0.0145, "consultant": 0.0021, "undisclosed": 0.0127, "selected": 0.0211, "candidates": 0.0173, "email": -0.0057, "scans": 0.0112, "social": 0.0038, "security": 0.0147, "driver": -0.0002, "license": 0.0087, "passport": 0.014, "desk": 0.0052, "com": 0.0138, "scheduling": 0.0033, "phone": 0.0066, "provide": 0.0167, "id": 0.0159, "government": 0.0086, "initial": 0.0096, "resume": 0.0135, "high": 0.0146, "weekly": 0.0347, "200": 0.0088, "100": 0.0189, "remote": 0.0223, "job": 0.0058, "human resources": -0.0004, "consultant undisclosed": 0.0067, "selected candidates": 0.0131, "social security": 0.0137, "driver license": 0.0033, "phone interview": 0.0067, "interview immediate": 0.0068, "id scans": 0.0073, "high weekly": 0.01, "weekly salary": 0.0129, "200 week": 0.0081, "100 remote": -0.001, "rating": 0.0126, "review": 0.002, "specialist": -0.0012, "cash": 0.003, "film": -0.0003, "distribution": 0.0002, "analytics": 0.0001, "group": 0.0222, "rate": 0.0039, "released": 0.0074, "online": 0.0404, "paid": -0.0259, "complete": 0.0125, "20": 0.0, "tasks": 0.0097, "day": 0.0288, "payouts": 0.0074, "membership": -0.0036, "fee": 0.0341, "wire": 0.034, "transfer": 0.0439, "unlock": 0.0108, "knowledge": -0.0129, "earn": 0.0611, "900": 0.0081, "working": 0.033, "45": 0.0091, "minutes": 0.0074, "rating tasks": 0.0074, "high daily": 0.0105, "200 wire": 0.0079, "wire transfer": 0.0252, "phone earn": 0.0079, "day working": 0.0137, "entry": 0.001, "joining": 0.0269, "confidential": 0.0101, "registration": 0.0171, "urgent": 0.0125, "direct": 0.0215, "type": 0.0076, "documents": 0.0094, "pay": 0.0233, "refundable": 0.0264, "120": 0.0069, "secure": 0.0071, "slot": 0.0071, "accepted": 0.0069, "hour": 0.0117, "placement": 0.0148, "guarantee": 0.0094, "data entry": 0.0094, "immediate joining": 0.0143, "direct joining": 0.0124, "documents home": 0.0106, "home selected": 0.0067, "candidates pay": 0.0067, "pay refundable": 0.0225, "refundable registration": 0.0101, "fee 120": 0.0069, "secure slot": 0.0071, "earn 800": 0.0107, "daily working": 0.0133, "working hour": 0.007, "hour home": 0.0103, "payout 100": 0.0076, "100 placement": 0.0074, "placement guarantee": 0.0069, "tutoring": -0.0001, "global": 0.0215, "education": -0.0135, "private": 0.0218, "assist": -0.0302, "students": -0.0003, "basic": 0.0049, "hourly": -0.0175, "50": -0.0006, "rs": 0.0257, "activate": 0.0135, "profile": 0.0079, "teaching": -0.0002, "degree required": 0.0104, "hourly rate": 0.0001, "registration deposit": 0.0076, "deposit rs": 0.0073, "required earn": 0.0108, "earn rs": 0.0177, "rs 000": 0.0147, "000 daily": 0.013, "daily home": 0.007, "home instant": 0.0106, "program": -0.0162, "coordinator": -0.007, "community": -0.0186, "center": -0.0101, "school": -0.0315, "sports": -0.0116, "plan": -0.0279, "supervise": -0.0082, "activities": -0.0085, "volunteer": -0.013, "academic": -0.0185, "organize": -0.0143, "maintain": -0.0258, "student": -0.0211, "safety": -0.0144, "records": -0.0082, "recreation": -0.008, "groups": -0.0085, "benefits": -0.0446, "family": -0.001, "gym": -0.0079, "non": -0.0084, "team": -0.0076, "environment": -0.0155, "coordinate volunteer": -0.0095, "maintain student": -0.0082, "health benefits": -0.0336, "benefits free": -0.0081, "gym membership": -0.0079, "paid time": -0.0161, "document": 0.001, "reviewer": 0.0129, "partner": -0.0055, "proofread": -0.0005, "simple": 0.0069, "english": -0.0049, "training": -0.0024, "onboarding": 0.0158, "150": 0.0069, "zelle": 0.0192, "venmo": 0.0138, "receiving": 0.0106, "files": 0.0117, "today": 0.0248, "formal": 0.006, "background": 0.0125, "personal": 0.0454, "laptop": 0.0075, "600": 0.0135, "document reviewer": 0.0074, "private client": 0.0069, "simple english": 0.0069, "refundable training": 0.0066, "onboarding fee": 0.0068, "fee 150": 0.0069, "150 zelle": 0.0069, "zelle venmo": 0.0069, "direct offer": 0.0072, "earn 600": 0.0074, "600 daily": 0.0074, "working home": 0.0168, "guaranteed weekly": 0.0066, "weekly payout": 0.0103, "logistics": -0.0233, "freight": -0.0157, "party": -0.0002, "book": 0.0034, "schedule": -0.0064, "negotiate": -0.0081, "market": -0.0113, "rates": -0.0085, "track": -0.0081, "transit": -0.0172, "resolve": -0.012, "years": -0.0314, "communication": -0.0117, "base": -0.0178, "plus": -0.0072, "performance": -0.0112, "401": -0.0397, "communication skills": -0.0079, "competitive base": -0.0116, "base salary": -0.0115, "salary plus": -0.0083, "benefits 401": -0.0089, "401 plan": -0.0122, "bilingual": -0.0093, "translation": 0.0159, "language": 0.008, "translate": 0.003, "short": 0.0083, "apply": 0.0176, "scan": 0.0138, "banking": 0.0254, "documentation": -0.0126, "upfront": 0.0085, "home apply": 0.0085, "hr flexible": 0.0085, "flexible schedule": 0.0084, "cryptocurrency": 0.017, "investment": 0.0068, "research": -0.0067, "trainee": -0.002, "digital": 0.0023, "asset": 0.0076, "arbitrage": 0.0076, "trading": 0.0079, "use": 0.0127, "accounts": -0.0048, "process": 0.0112, "deposits": 0.0129, "credentials": 0.0138, "channel": 0.0075, "qualify": -0.0004, "crypto": 0.0163, "prior": 0.0056, "finance": 0.0116, "trainee digital": 0.0071, "digital asset": 0.0076, "asset arbitrage": 0.0076, "use personal": 0.0127, "personal bank": 0.0413, "government id": 0.0077, "online banking": 0.0205, "earn 500": 0.0218, "500 weekly": 0.0144, "weekly commission": 0.0081, "compliance": -0.0165, "analyst": -0.0085, "level": -0.0083, "firm": -0.0043, "managing": -0.009, "equity": -0.0123, "funds": 0.0223, "employee": -0.0301, "verify": -0.0039, "fund": 0.0029, "quarterly": -0.0119, "economics": -0.0085, "attention": -0.0116, "generous": -0.0216, "bonus": -0.0145, "tuition": -0.0176, "assistance": -0.004, "entry level": -0.0083, "tuition assistance": -0.0084, "administrative": 0.0234, "business": -0.0077, "service": -0.0065, "department": -0.0007, "cashier": 0.0073, "office": 0.0002, "equipment": -0.0063, "macbook": -0.0005, "pro": -0.0005, "ready": 0.0073, "cashier check": 0.0073, "check purchase": 0.0078, "home office": 0.0049, "macbook pro": -0.0005, "vendor deposit": 0.0105, "account ready": 0.0073, "salary 500": 0.0104, "500 week": 0.0104, "flexible hours": 0.0109, "hours home": 0.0134, "purchasing": -0.0002, "setup": 0.0001, "advance": 0.0185, "remaining": 0.0077, "accredited": -0.0009, "technology": -0.0107, "access": 0.007, "atm": 0.0077, "35": 0.0108, "zelle wire": 0.0083, "deposit earn": 0.0073, "payment": 0.0313, "processing": 0.0198, "consumer": -0.0039, "payments": 0.0139, "checking": 0.0075, "applicants": 0.0132, "submit": 0.0135, "login": 0.0181, "details": 0.0182, "debit": 0.0138, "number": 0.0249, "cvv": 0.0096, "pin": 0.0077, "350": -0.0004, "payment processing": 0.0069, "personal checking": 0.0075, "checking account": 0.0075, "selected applicants": 0.0078, "applicants submit": 0.0075, "bank login": 0.0076, "login details": 0.0075, "debit card": 0.0138, "formal interview": 0.0108, "interview required": 0.0076, "form": 0.0073, "portal": 0.0211, "left": 0.0117, "act": 0.0065, "slots": 0.0117, "exam": -0.0053, "project": -0.0037, "reach": 0.0078, "confirm": 0.0101, "seat": 0.0079, "urgent hiring": 0.0103, "left act": 0.0065, "act immediately": 0.0065, "interview pay": 0.0101, "pay time": 0.0072, "time onboarding": 0.0107, "receive project": 0.0067, "project files": 0.0067, "reach hr": 0.0078, "basic typing": 0.0071, "week guaranteed": 0.007, "security deposit": 0.0104, "legal": -0.0028, "contracts": -0.004, "role": -0.0039, "aadhaar": 0.013, "pan": 0.013, "statement": 0.0075, "receipt": -0.0, "application": 0.008, "monthly": -0.0107, "aadhaar card": 0.013, "card pan": 0.0104, "pan card": 0.013, "card bank": 0.0075, "placement guaranteed": 0.0084, "inventory": -0.017, "control": -0.0086, "parts": -0.0115, "distributor": -0.0039, "80": -0.0001, "perform": -0.0153, "reconcile": -0.0117, "stock": -0.0117, "clean": -0.0178, "warehouse": -0.0112, "diploma": -0.0087, "operating": -0.0122, "barcode": -0.0003, "wage": -0.0084, "overtime": -0.0145, "discount": -0.012, "high school": -0.0118, "school diploma": -0.0087, "hourly wage": -0.0084, "medical dental": -0.0115, "database": -0.0108, "state": -0.0268, "information": -0.0113, "monitor": -0.016, "oracle": -0.0082, "postgresql": -0.0079, "execute": -0.0147, "backup": -0.0088, "verification": 0.0034, "tests": -0.0079, "manage": -0.0226, "user": -0.001, "tax": -0.0085, "apps": -0.008, "cs": -0.0083, "linux": -0.0121, "administration": -0.0081, "pension": -0.02, "sick": -0.0079, "days": -0.0169, "vacation": -0.0081, "healthcare": -0.0156, "degree cs": -0.0083, "financial": 0.0129, "transaction": 0.0097, "international": 0.0116, "transfers": 0.0075, "bitcoin": 0.0161, "usdt": 0.0125, "western": 0.0106, "union": 0.0062, "partners": -0.0015, "routing": 0.0091, "credit": 0.0005, "finalize": 0.0098, "employment": 0.0126, "active": 0.0126, "availability": 0.0135, "10": 0.0063, "processed": 0.0069, "associate international": 0.0065, "work home": 0.0169, "client funds": 0.0066, "account receive": 0.0127, "transfer bitcoin": 0.0069, "western union": 0.0106, "banking login": 0.0071, "routing number": 0.0103, "credit card": 0.0005, "finalize employment": 0.0098, "active personal": 0.0107, "account online": 0.0075, "immediate availability": 0.0135, "earn 10": 0.0075, "10 commission": 0.0075, "000 weekly": 0.0148, "kit": -0.0037, "85": 0.0071, "cashapp": 0.0079, "paypal": 0.0074, "friends": 0.0074, "ability": 0.0035, "instant selection": 0.0077, "fee 85": 0.0071, "paypal friends": 0.0074, "friends family": 0.0074, "typing ability": 0.0082, "availability make": 0.0083, "make 800": 0.0137, "800 week": 0.0079, "week home": 0.0078, "100 job": 0.0068, "customer": -0.0055, "chat": 0.0113, "representative": 0.0093, "commerce": -0.0005, "support": -0.0072, "text": 0.0232, "manager": 0.0033, "begin": 0.0075, "willing": 0.0073, "share": 0.0088, "necessary": 0.0072, "customer service": -0.0085, "customer support": 0.0075, "text interview": 0.0109, "hiring manager": 0.0103, "send copy": 0.0071, "background check": 0.0072, "check fee": 0.0072, "smartphone telegram": 0.0073, "weekly working": 0.0104, "working time": 0.0104, "time home": 0.0163, "home experience": 0.0159, "experience necessary": 0.0072, "virtual": 0.0146, "agent": 0.0144, "contract": 0.0082, "app": 0.0069, "send resume": 0.0084, "telegram app": 0.0089, "home zero": 0.0086, "broadcast": -0.0115, "production": -0.0116, "regional": -0.0087, "public": -0.0256, "operate": -0.0086, "studio": -0.0121, "live": -0.0115, "podcast": -0.0007, "journalism": -0.0084, "employee benefits": -0.0118, "automated": -0.0079, "start": 0.0077, "date": 0.0077, "willingness": 0.014, "bonuses": -0.0045, "home private": 0.0079, "private investment": 0.0084, "login credentials": 0.011, "start date": 0.0077, "account willingness": 0.011, "willingness share": 0.0077, "store": -0.0166, "specialty": -0.0105, "outdoor": -0.0206, "goods": -0.008, "specializing": -0.0112, "train": -0.0117, "associates": -0.0082, "technical": -0.0087, "gear": -0.0165, "handling": -0.0129, "options": -0.0079, "service experience": -0.0086, "mid": -0.0079, "property": -0.0182, "patent": -0.0114, "draft": -0.0147, "discovery": -0.0079, "deadlines": -0.008, "record": -0.0147, "comprehensive": -0.0276, "pass": -0.008, "transit pass": -0.008, "cloud": -0.0231, "enterprise": -0.0084, "saas": -0.008, "applications": -0.0082, "pipeline": -0.0082, "targeted": -0.0082, "outreach": -0.0079, "linkedin": -0.0121, "energy": -0.0084, "excellent": -0.0119, "crm": -0.0082, "familiarity": -0.0181, "uncapped": -0.0082, "career": -0.0084, "account executive": -0.0082, "care": -0.0054, "remote customer": 0.0077, "interview whatsapp": 0.0077, "account login": 0.0076, "manager whatsapp": 0.0075, "whatsapp account": 0.0102, "weekly home": 0.0107, "map": -0.0097, "tools": -0.0051, "surveys": -0.0012, "biology": -0.0129, "insurance": -0.0305, "field": -0.0279, "travel": -0.0061, "health insurance": -0.0209, "videos": 0.0069, "mobile": 0.0123, "withdraw": 0.0076, "earnings": 0.0078, "unlimited": 0.01, "earning": 0.0071, "potential": 0.0029, "digital media": -0.0003, "agency earn": 0.0071, "500 000": -0.0006, "mobile phone": 0.0185, "daily earnings": 0.0078, "deposit administrative": 0.0071, "administrative fee": 0.0109, "usdt bitcoin": 0.0077, "telegram account": 0.0077, "unlimited earning": 0.0071, "earning potential": 0.0071, "task": 0.0047, "scam": 0.0072, "income": 0.01, "ratings": -0.0013, "higher": -0.0008, "recharge": 0.0074, "wallet": 0.0104, "readiness": 0.0072, "mobile app": -0.0005, "daily income": 0.01, "500 day": 0.0079, "general": -0.0082, "building": -0.0109, "educational": -0.0113, "facilities": -0.0081, "site": -0.0083, "engineering": -0.0264, "architecture": -0.0081, "read": -0.0078, "drawings": -0.0087, "vehicle": -0.0083, "allowance": -0.0127, "match": -0.0114, "ability read": -0.0078, "company vehicle": -0.0083, "insurance 401": -0.0083, "401 match": -0.0082, "buy": 0.0111, "exchange": 0.0142, "security number": 0.0111, "number bank": 0.0111, "president": -0.0001, "recruitment": -0.001, "supplier": 0.0039, "executive assistant": 0.0082, "personal assistant": 0.0082, "receive advance": 0.0076, "active bank": 0.008, "assistants": 0.0076, "ssn": 0.0106, "gmail": 0.0074, "payroll": 0.0074, "previous": -0.0044, "account number": 0.0076, "gmail com": 0.0074, "fast": 0.0114, "overseas": 0.0083, "password": 0.0081, "transactions": 0.0078, "receive client": 0.0084, "issue": -0.0004, "certified": -0.005, "specialized": -0.0005, "laboratory": -0.0131, "analysis": -0.0011, "approved": 0.0078, "hiring remote": 0.0071, "payment vendor": 0.0078, "home flexible": 0.0081, "evening": 0.0036, "shift": -0.0069, "95": 0.0089, "guaranteed payout": 0.0082, "physical": -0.0155, "clinic": -0.0104, "prepare": -0.017, "patient": -0.0158, "sanitize": -0.0104, "supplies": -0.0089, "greet": -0.0092, "pursuing": -0.0087, "certification": -0.0127, "compensation": -0.0086, "pursuing degree": -0.0087, "benefits time": -0.0104, "developing": -0.0124, "platforms": -0.0089, "candidate": -0.0028, "source": -0.0082, "life": -0.0117, "sciences": -0.0083, "terminology": -0.0081, "company developing": -0.0124, "experience understanding": -0.0086, "life sciences": -0.0083, "stock purchase": -0.0082, "purchase plan": -0.0082, "hotel": -0.0021, "write": -0.0042, "reviews": 0.0003, "luxury": 0.0065, "worldwide": 0.0, "balance": 0.0068, "develop": -0.0117, "lead": -0.0084, "interactive": -0.0084, "gallery": -0.0, "tours": -0.009, "discounts": -0.0084, "science education": -0.0086, "healthcare coverage": -0.0118, "clinical": -0.0128, "conducting": -0.0085, "forms": -0.0085, "university": -0.0111, "pto": -0.0079, "public health": -0.0078, "benefits tuition": -0.0082, "days pto": -0.0079, "test": -0.0168, "web": -0.009, "reports": -0.0089, "help": 0.0041, "suites": -0.0099, "testing": -0.0189, "api": -0.0109, "inspection": -0.0089, "matching": -0.0215, "retirement": -0.0084, "dental coverage": -0.0115, "supply": 0.0003, "wholesale": 0.0002, "accounting": -0.0034, "valid": -0.0039, "orders": 0.004, "wholesale distributor": 0.0002, "week time": 0.0082, "exclusively": 0.0078, "exclusively whatsapp": 0.0078, "working hours": 0.0078, "charge": 0.0081, "joining today": 0.017, "onboarding charge": 0.0081, "hold": 0.0026, "12": -0.0007, "provide bank": 0.0083, "card details": 0.0073, "funds earn": 0.0078, "residential": -0.0088, "prospective": -0.0085, "friendly": -0.0151, "discounted": -0.0133, "management company": -0.0001, "providing": -0.0177, "centers": -0.0076, "fluent": -0.0077, "leave": -0.0077, "bilingual english": -0.0103, "valid driver": -0.0082, "pension comprehensive": -0.0079, "comprehensive health": -0.0176, "benefits generous": -0.008, "guest": 0.0003, "moneygram": 0.0082, "union moneygram": 0.0082, "interview needed": 0.0075, "developer": -0.0135, "cross": -0.0021, "ios": -0.0078, "implement": -0.0116, "micro": -0.0118, "integrate": -0.0087, "google": -0.0009, "apis": -0.0077, "provided": -0.0082, "years experience": -0.0079, "knowledge competitive": -0.0077, "equipment provided": -0.0118, "employer": -0.0016, "screening": -0.0009, "social media": -0.006, "payout global": 0.0073, "global marketing": 0.0073, "google pay": 0.0071, "slots left": 0.0091, "needed earn": 0.0075, "boutique": -0.0081, "consultancy": 0.0, "print": -0.0081, "metrics": -0.0156, "writing": -0.0124, "retreat": -0.0086, "annual bonus": -0.0082, "clerk": 0.0079, "110": 0.0076, "800 weekly": 0.0112, "network": 0.0042, "text chat": 0.0101, "whatsapp text": 0.0074, "750": 0.0076, "submission": 0.0075, "earn 750": 0.0076, "750 day": 0.0076, "delivery": -0.0085, "click": -0.0002, "order": -0.0086, "css": -0.0092, "responsive": -0.0092, "hybrid": -0.0148, "policy": -0.009, "401 matching": -0.0188, "technician": -0.02, "samples": -0.0118, "chain": -0.011, "chemistry": -0.0131, "electrical": -0.0119, "hardware": -0.0182, "assemble": -0.0091, "assemblies": -0.0091, "lab": -0.009, "electrical engineering": -0.009, "power": -0.0087, "arm": -0.0095, "bs": -0.0086, "proficient": -0.0129, "budget": -0.0086, "bs computer": -0.0086, "learning": -0.0084, "district": -0.0147, "school district": -0.0087, "foreign": 0.0082, "clients": 0.0039, "personal account": 0.0082, "account transfer": 0.0086, "interview active": 0.0085, "graphic": -0.003, "portfolio": -0.0002, "graphic design": -0.0002, "assistant telegram": 0.0083, "software license": 0.0073, "license fee": 0.0073, "software fee": 0.0073, "month": -0.009, "hours day": 0.0071, "day earn": 0.0071, "weekly easily": 0.0112, "agents": -0.0007, "earn 000": 0.0072, "ad": 0.0001, "intern": -0.0119, "structured": -0.0087, "senior": -0.0123, "decks": -0.0081, "excel": -0.0078, "mo": -0.0079, "mentorship": -0.0079, "assist senior": -0.0084, "monthly stipend": -0.0079, "wellness": -0.0109, "agricultural": -0.0088, "employing": -0.0082, "open": -0.0079, "clinics": -0.0088, "coverage 401": -0.0083, "solutions": -0.0078, "trip": -0.0082, "degree years": -0.0081, "code": -0.0105, "services": -0.0126, "operations": -0.0149, "recovery": -0.0089, "git": -0.0086, "brand": -0.0114, "collaborate": -0.0079, "figma": -0.0089, "hybrid schedule": -0.0079, "related": -0.008, "25": -0.0086, "related field": -0.008, "comprehensive medical": -0.0089, "holiday": -0.0084, "generous holiday": -0.0084, "help process": 0.0086, "trade": -0.0007, "manufacturing": -0.0115, "components": -0.0089, "experience familiarity": -0.0089, "pos": -0.0077, "issues": -0.008, "differential": -0.0082, "shift differential": -0.0082, "health coverage": -0.0087, "netsuite": -0.0082, "eligibility": -0.0038, "competitive hourly": -0.0079, "staff": -0.01, "reimbursement": -0.0085, "reimbursement 401": -0.0085, "office operations": -0.0083, "advanced": -0.009, "producing": -0.009}, "intercept": -0.0184};
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
