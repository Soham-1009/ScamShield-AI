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
  const ML_MODEL = {"stop": ["a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"], "idf": {"quality": 3.783, "scheme": 4.4761, "commercial": 3.783, "retail": 3.6288, "agency": 3.0098, "receive": 2.7415, "shopping": 4.4761, "assignments": 4.1884, "mail": 4.4761, "send": 2.8667, "check": 3.1768, "800": 3.2721, "deposit": 2.7415, "bank": 2.4392, "500": 2.8667, "commission": 3.1768, "purchase": 3.3775, "apple": 4.4761, "gift": 4.1884, "cards": 4.4761, "local": 3.9653, "stores": 4.4761, "photos": 4.4761, "card": 2.8021, "codes": 4.1884, "telegram": 3.2721, "recruiter": 3.783, "account": 2.1407, "company": 2.7415, "checks": 4.4761, "smartphone": 3.4953, "camera": 4.4761, "assignment": 4.1884, "flexible": 3.1768, "time": 2.4837, "hours": 2.8021, "send check": 4.4761, "check 800": 4.4761, "800 deposit": 4.4761, "deposit check": 3.6288, "check bank": 4.4761, "send photos": 4.4761, "gift card": 4.4761, "bank account": 2.5302, "flexible time": 4.1884, "product": 3.9653, "marketing": 3.6288, "associate": 3.0898, "management": 3.2721, "software": 3.0098, "vendor": 3.1768, "creative": 3.783, "design": 3.4953, "conduct": 3.4953, "feature": 4.4761, "create": 4.4761, "sales": 3.783, "coordinate": 2.9357, "release": 4.4761, "interview": 2.3558, "video": 3.783, "case": 4.1884, "studies": 4.1884, "bachelor": 2.6843, "degree": 2.3967, "communications": 3.9653, "strong": 3.1768, "skills": 2.7415, "analytical": 4.4761, "competitive": 2.9357, "salary": 2.8021, "annual": 3.4953, "professional": 4.4761, "development": 4.4761, "stipend": 3.6288, "health": 2.2789, "dental": 3.2721, "management software": 4.4761, "software vendor": 3.9653, "creative design": 4.4761, "bachelor degree": 3.2721, "communications strong": 4.4761, "competitive salary": 3.6288, "salary annual": 4.4761, "health dental": 3.783, "typing": 3.783, "copy": 3.783, "paste": 4.4761, "executive": 3.6288, "leading": 4.1884, "multinational": 4.4761, "client": 3.0898, "hiring": 3.0098, "immediately": 3.0898, "work": 2.6303, "daily": 2.5302, "contact": 3.6288, "hr": 3.0898, "whatsapp": 3.2721, "instant": 3.1768, "offer": 3.2721, "letter": 3.4953, "zero": 3.9653, "experience": 1.8611, "needed": 3.0898, "required": 3.0898, "make": 3.0898, "000": 2.6303, "week": 2.9357, "easily": 3.6288, "home": 1.9111, "guaranteed": 2.3558, "payout": 3.0898, "copy paste": 4.4761, "work hours": 3.9653, "contact hr": 4.4761, "hr whatsapp": 4.1884, "instant offer": 4.1884, "offer letter": 3.4953, "zero experience": 3.9653, "experience needed": 3.4953, "make 000": 3.783, "000 week": 3.9653, "easily home": 4.1884, "home guaranteed": 3.0898, "guaranteed daily": 3.6288, "daily payout": 3.783, "junior": 3.6288, "data": 3.3775, "engineer": 3.783, "streaming": 4.4761, "media": 3.4953, "platform": 3.6288, "delivering": 4.4761, "audio": 4.1884, "content": 4.1884, "build": 3.6288, "optimize": 3.9653, "batch": 4.4761, "real": 4.1884, "using": 2.9357, "structure": 4.4761, "computer": 3.6288, "science": 3.3775, "python": 3.9653, "sql": 4.1884, "understanding": 3.9653, "concepts": 4.1884, "free": 3.4953, "premium": 4.4761, "medical": 3.0898, "coverage": 3.0098, "internet": 4.1884, "subsidy": 4.1884, "audio video": 4.4761, "real time": 4.1884, "computer science": 4.4761, "salary medical": 4.4761, "medical coverage": 4.1884, "human": 4.4761, "resources": 4.4761, "assistant": 2.8021, "immediate": 3.0098, "selection": 3.9653, "corporate": 3.4953, "consultant": 4.1884, "undisclosed": 3.9653, "selected": 3.4953, "candidates": 3.9653, "email": 3.6288, "scans": 4.4761, "social": 3.3775, "security": 3.3775, "driver": 3.6288, "license": 3.4953, "passport": 4.1884, "desk": 3.783, "com": 3.9653, "scheduling": 4.1884, "phone": 2.9357, "provide": 3.4953, "id": 3.783, "government": 3.4953, "initial": 3.783, "resume": 3.9653, "high": 2.7415, "weekly": 2.5302, "200": 3.6288, "100": 3.2721, "remote": 2.6843, "job": 3.9653, "human resources": 4.4761, "consultant undisclosed": 4.4761, "selected candidates": 3.9653, "social security": 3.9653, "driver license": 3.783, "phone interview": 4.4761, "interview immediate": 4.4761, "id scans": 4.4761, "high weekly": 4.1884, "weekly salary": 3.9653, "200 week": 4.4761, "100 remote": 4.4761, "rating": 4.4761, "review": 3.783, "specialist": 3.1768, "cash": 3.783, "film": 4.4761, "distribution": 3.9653, "analytics": 4.4761, "group": 3.4953, "rate": 4.1884, "released": 4.4761, "online": 2.8021, "paid": 2.8667, "complete": 3.6288, "20": 4.4761, "tasks": 4.4761, "day": 3.2721, "payouts": 4.4761, "membership": 4.1884, "fee": 2.8021, "wire": 3.0098, "transfer": 2.6843, "unlock": 4.1884, "knowledge": 3.3775, "earn": 2.0782, "900": 4.4761, "working": 2.7415, "45": 3.783, "minutes": 4.4761, "rating tasks": 4.4761, "high daily": 4.1884, "200 wire": 4.4761, "wire transfer": 3.2721, "phone earn": 4.4761, "day working": 3.9653, "entry": 3.9653, "joining": 3.2721, "confidential": 4.1884, "registration": 3.783, "urgent": 3.9653, "direct": 3.3775, "type": 4.4761, "documents": 3.783, "pay": 2.6303, "refundable": 3.1768, "120": 4.4761, "secure": 4.4761, "slot": 4.4761, "accepted": 4.4761, "hour": 3.6288, "placement": 3.4953, "guarantee": 4.1884, "data entry": 4.4761, "immediate joining": 3.9653, "direct joining": 3.9653, "documents home": 4.1884, "home selected": 4.4761, "candidates pay": 4.4761, "pay refundable": 3.3775, "refundable registration": 4.1884, "fee 120": 4.4761, "secure slot": 4.4761, "earn 800": 4.1884, "daily working": 3.9653, "working hour": 4.4761, "hour home": 4.1884, "payout 100": 4.4761, "100 placement": 4.4761, "placement guarantee": 4.4761, "tutoring": 4.4761, "global": 3.1768, "education": 3.783, "private": 3.1768, "assist": 2.8021, "students": 4.4761, "basic": 3.3775, "hourly": 3.4953, "50": 3.9653, "rs": 3.6288, "activate": 3.9653, "profile": 4.4761, "teaching": 4.4761, "degree required": 4.4761, "hourly rate": 4.4761, "registration deposit": 4.4761, "deposit rs": 4.4761, "required earn": 4.1884, "earn rs": 3.6288, "rs 000": 4.1884, "000 daily": 3.9653, "daily home": 4.4761, "home instant": 4.1884, "program": 3.9653, "coordinator": 3.1768, "community": 3.9653, "center": 3.783, "school": 3.2721, "sports": 4.4761, "plan": 3.2721, "supervise": 4.4761, "activities": 4.4761, "volunteer": 4.1884, "academic": 3.9653, "organize": 3.9653, "maintain": 3.3775, "student": 3.9653, "safety": 4.1884, "records": 4.4761, "recreation": 4.4761, "groups": 4.4761, "benefits": 2.579, "family": 3.9653, "gym": 4.4761, "non": 4.4761, "team": 3.6288, "environment": 3.9653, "coordinate volunteer": 4.4761, "maintain student": 4.4761, "health benefits": 3.0898, "benefits free": 4.4761, "gym membership": 4.4761, "paid time": 3.9653, "document": 3.6288, "reviewer": 3.9653, "partner": 3.783, "proofread": 4.4761, "simple": 4.4761, "english": 3.6288, "training": 3.3775, "onboarding": 3.783, "150": 4.4761, "zelle": 3.6288, "venmo": 3.9653, "receiving": 4.1884, "files": 3.6288, "today": 3.3775, "formal": 3.783, "background": 3.9653, "personal": 2.579, "laptop": 4.4761, "600": 3.9653, "document reviewer": 4.4761, "private client": 4.4761, "simple english": 4.4761, "refundable training": 4.4761, "onboarding fee": 4.4761, "fee 150": 4.4761, "150 zelle": 4.4761, "zelle venmo": 4.4761, "direct offer": 4.4761, "earn 600": 4.4761, "600 daily": 4.4761, "working home": 3.783, "guaranteed weekly": 4.4761, "weekly payout": 4.1884, "logistics": 3.9653, "freight": 4.1884, "party": 4.4761, "book": 4.4761, "schedule": 3.6288, "negotiate": 4.4761, "market": 4.1884, "rates": 4.4761, "track": 4.4761, "transit": 3.9653, "resolve": 4.1884, "years": 3.0898, "communication": 4.1884, "base": 3.783, "plus": 3.6288, "performance": 4.1884, "401": 2.8667, "communication skills": 4.4761, "competitive base": 4.1884, "base salary": 4.1884, "salary plus": 4.4761, "benefits 401": 4.4761, "401 plan": 4.1884, "bilingual": 4.1884, "translation": 4.4761, "language": 4.4761, "translate": 4.1884, "short": 4.4761, "apply": 3.783, "scan": 3.9653, "banking": 3.3775, "documentation": 3.6288, "upfront": 4.4761, "home apply": 4.4761, "hr flexible": 4.4761, "flexible schedule": 4.4761, "cryptocurrency": 3.9653, "investment": 3.9653, "research": 3.783, "trainee": 3.6288, "digital": 3.1768, "asset": 4.4761, "arbitrage": 4.4761, "trading": 4.1884, "use": 3.9653, "accounts": 4.1884, "process": 3.1768, "deposits": 3.9653, "credentials": 3.9653, "channel": 4.4761, "qualify": 4.4761, "crypto": 3.783, "prior": 3.9653, "finance": 3.6288, "trainee digital": 4.4761, "digital asset": 4.4761, "asset arbitrage": 4.4761, "use personal": 3.9653, "personal bank": 2.8667, "government id": 4.4761, "online banking": 3.6288, "earn 500": 3.4953, "500 weekly": 3.9653, "weekly commission": 4.4761, "compliance": 4.1884, "analyst": 4.4761, "level": 4.4761, "firm": 3.783, "managing": 4.4761, "equity": 4.1884, "funds": 3.2721, "employee": 3.2721, "verify": 3.783, "fund": 4.1884, "quarterly": 4.1884, "economics": 4.4761, "attention": 4.1884, "generous": 3.6288, "bonus": 3.4953, "tuition": 3.783, "assistance": 4.1884, "entry level": 4.4761, "tuition assistance": 4.4761, "administrative": 3.4953, "business": 3.6288, "service": 3.2721, "department": 4.4761, "cashier": 4.4761, "office": 3.4953, "equipment": 3.3775, "macbook": 4.4761, "pro": 4.4761, "ready": 4.4761, "cashier check": 4.4761, "check purchase": 4.4761, "home office": 4.1884, "macbook pro": 4.4761, "vendor deposit": 4.1884, "account ready": 4.4761, "salary 500": 4.1884, "500 week": 4.1884, "flexible hours": 4.1884, "hours home": 3.9653, "purchasing": 4.4761, "setup": 3.9653, "advance": 3.783, "remaining": 4.4761, "accredited": 4.4761, "technology": 3.783, "access": 4.4761, "atm": 4.4761, "35": 4.1884, "zelle wire": 4.4761, "deposit earn": 4.4761, "payment": 3.0898, "processing": 3.6288, "consumer": 4.1884, "payments": 4.1884, "checking": 4.4761, "applicants": 3.9653, "submit": 3.9653, "login": 3.6288, "details": 3.6288, "debit": 3.9653, "number": 3.6288, "cvv": 4.1884, "pin": 4.4761, "350": 4.4761, "payment processing": 4.4761, "personal checking": 4.4761, "checking account": 4.4761, "selected applicants": 4.4761, "applicants submit": 4.4761, "bank login": 4.4761, "login details": 4.4761, "debit card": 3.9653, "formal interview": 4.1884, "interview required": 4.4761, "form": 4.4761, "portal": 3.6288, "left": 4.1884, "act": 4.4761, "slots": 4.1884, "exam": 4.1884, "project": 3.9653, "reach": 4.4761, "confirm": 4.1884, "seat": 4.4761, "urgent hiring": 4.1884, "left act": 4.4761, "act immediately": 4.4761, "interview pay": 4.1884, "pay time": 4.4761, "time onboarding": 4.1884, "receive project": 4.4761, "project files": 4.4761, "reach hr": 4.4761, "basic typing": 4.4761, "week guaranteed": 4.4761, "security deposit": 4.1884, "legal": 4.4761, "contracts": 4.1884, "role": 4.1884, "aadhaar": 3.9653, "pan": 3.9653, "statement": 4.4761, "receipt": 4.4761, "application": 4.4761, "monthly": 3.783, "aadhaar card": 3.9653, "card pan": 4.1884, "pan card": 3.9653, "card bank": 4.4761, "placement guaranteed": 4.4761, "inventory": 3.9653, "control": 4.4761, "parts": 4.4761, "distributor": 4.1884, "80": 4.4761, "perform": 3.9653, "reconcile": 4.1884, "stock": 4.1884, "clean": 3.783, "warehouse": 4.4761, "diploma": 3.9653, "operating": 4.1884, "barcode": 4.4761, "wage": 4.4761, "overtime": 3.9653, "discount": 4.1884, "high school": 3.783, "school diploma": 3.9653, "hourly wage": 4.4761, "medical dental": 4.1884, "database": 4.4761, "state": 3.6288, "information": 4.1884, "monitor": 3.9653, "oracle": 4.4761, "postgresql": 4.4761, "execute": 3.9653, "backup": 4.4761, "verification": 4.1884, "tests": 4.4761, "manage": 3.4953, "user": 4.4761, "tax": 4.4761, "apps": 4.4761, "cs": 4.4761, "linux": 4.4761, "administration": 4.4761, "pension": 3.6288, "sick": 4.4761, "days": 3.9653, "vacation": 4.4761, "healthcare": 3.3775, "degree cs": 4.4761, "financial": 3.9653, "transaction": 4.4761, "international": 4.1884, "transfers": 4.4761, "bitcoin": 3.783, "usdt": 3.9653, "western": 4.1884, "union": 3.9653, "partners": 4.4761, "routing": 3.783, "credit": 3.9653, "finalize": 4.1884, "employment": 3.9653, "active": 3.6288, "availability": 3.9653, "10": 4.1884, "processed": 4.4761, "associate international": 4.4761, "work home": 3.783, "client funds": 4.4761, "account receive": 3.9653, "transfer bitcoin": 4.4761, "western union": 4.1884, "banking login": 4.4761, "routing number": 4.1884, "credit card": 3.9653, "finalize employment": 4.1884, "active personal": 4.1884, "account online": 4.4761, "immediate availability": 3.9653, "earn 10": 4.4761, "10 commission": 4.4761, "000 weekly": 3.783, "kit": 4.4761, "85": 4.4761, "cashapp": 4.4761, "paypal": 4.4761, "friends": 4.4761, "ability": 3.783, "instant selection": 4.4761, "fee 85": 4.4761, "paypal friends": 4.4761, "friends family": 4.4761, "typing ability": 4.4761, "availability make": 4.4761, "make 800": 3.9653, "800 week": 4.4761, "week home": 4.4761, "100 job": 4.4761, "customer": 3.4953, "chat": 3.783, "representative": 3.783, "commerce": 4.4761, "support": 3.783, "text": 3.4953, "manager": 3.6288, "begin": 4.4761, "willing": 4.4761, "share": 3.9653, "necessary": 4.4761, "customer service": 3.9653, "customer support": 4.4761, "text interview": 4.1884, "hiring manager": 4.1884, "send copy": 4.4761, "background check": 4.4761, "check fee": 4.4761, "smartphone telegram": 4.4761, "weekly working": 4.1884, "working time": 4.1884, "time home": 3.783, "home experience": 3.783, "experience necessary": 4.4761, "virtual": 4.1884, "agent": 3.9653, "contract": 4.4761, "app": 3.9653, "send resume": 4.4761, "telegram app": 4.4761, "home zero": 4.4761, "broadcast": 4.4761, "production": 4.1884, "regional": 3.6288, "public": 3.3775, "operate": 4.4761, "studio": 3.9653, "live": 4.1884, "podcast": 4.4761, "journalism": 4.4761, "employee benefits": 4.1884, "automated": 3.9653, "start": 4.4761, "date": 4.4761, "willingness": 3.9653, "bonuses": 4.1884, "home private": 4.4761, "private investment": 4.4761, "login credentials": 4.1884, "start date": 4.4761, "account willingness": 4.1884, "willingness share": 4.4761, "store": 4.1884, "specialty": 4.4761, "outdoor": 4.1884, "goods": 4.4761, "specializing": 4.1884, "train": 4.1884, "associates": 4.4761, "technical": 3.783, "gear": 4.1884, "handling": 4.1884, "options": 4.4761, "service experience": 4.4761, "mid": 4.4761, "property": 4.1884, "patent": 4.4761, "draft": 3.9653, "discovery": 4.4761, "deadlines": 4.4761, "record": 4.1884, "comprehensive": 3.2721, "pass": 4.4761, "transit pass": 4.4761, "cloud": 3.9653, "enterprise": 4.4761, "saas": 4.4761, "applications": 3.9653, "pipeline": 4.4761, "targeted": 4.4761, "outreach": 4.4761, "linkedin": 4.1884, "energy": 4.4761, "excellent": 4.1884, "crm": 4.4761, "familiarity": 3.783, "uncapped": 4.4761, "career": 4.4761, "account executive": 4.4761, "care": 4.1884, "remote customer": 4.4761, "interview whatsapp": 4.4761, "account login": 4.4761, "manager whatsapp": 4.4761, "whatsapp account": 4.1884, "weekly home": 4.1884, "map": 4.4761, "tools": 4.1884, "surveys": 4.4761, "biology": 4.1884, "insurance": 3.2721, "field": 3.4953, "travel": 3.6288, "health insurance": 3.6288, "videos": 4.4761, "mobile": 3.1768, "withdraw": 4.4761, "earnings": 4.4761, "unlimited": 4.1884, "earning": 4.4761, "potential": 4.1884, "digital media": 4.4761, "agency earn": 4.4761, "500 000": 4.4761, "mobile phone": 3.6288, "daily earnings": 4.4761, "deposit administrative": 4.4761, "administrative fee": 4.1884, "usdt bitcoin": 4.4761, "telegram account": 4.4761, "unlimited earning": 4.4761, "earning potential": 4.4761, "task": 4.1884, "scam": 4.4761, "income": 4.1884, "ratings": 4.4761, "higher": 4.4761, "recharge": 4.4761, "wallet": 4.1884, "readiness": 4.4761, "mobile app": 4.4761, "daily income": 4.1884, "500 day": 4.4761, "general": 4.4761, "building": 4.1884, "educational": 4.1884, "facilities": 4.4761, "site": 4.4761, "engineering": 3.6288, "architecture": 4.4761, "read": 4.4761, "drawings": 4.4761, "vehicle": 4.4761, "allowance": 4.1884, "match": 4.1884, "ability read": 4.4761, "company vehicle": 4.4761, "insurance 401": 4.4761, "401 match": 4.4761, "buy": 4.1884, "exchange": 4.1884, "security number": 4.1884, "number bank": 4.1884, "president": 4.4761, "recruitment": 3.9653, "supplier": 4.1884, "executive assistant": 4.4761, "personal assistant": 4.4761, "receive advance": 4.4761, "active bank": 4.4761, "assistants": 4.4761, "ssn": 4.4761, "gmail": 4.4761, "payroll": 4.4761, "previous": 4.1884, "account number": 4.4761, "gmail com": 4.4761, "fast": 4.1884, "overseas": 4.4761, "password": 4.4761, "transactions": 4.4761, "receive client": 4.4761, "issue": 4.4761, "certified": 4.1884, "specialized": 4.4761, "laboratory": 3.9653, "analysis": 4.4761, "approved": 4.4761, "hiring remote": 4.4761, "payment vendor": 4.4761, "home flexible": 4.4761, "evening": 4.4761, "shift": 3.9653, "95": 4.4761, "guaranteed payout": 4.4761, "physical": 4.4761, "clinic": 4.4761, "prepare": 3.9653, "patient": 3.9653, "sanitize": 4.4761, "supplies": 4.4761, "greet": 4.4761, "pursuing": 4.4761, "certification": 4.1884, "compensation": 4.4761, "pursuing degree": 4.4761, "benefits time": 4.4761, "developing": 4.1884, "platforms": 4.4761, "candidate": 4.4761, "source": 4.4761, "life": 4.1884, "sciences": 4.4761, "terminology": 4.4761, "company developing": 4.1884, "experience understanding": 4.4761, "life sciences": 4.4761, "stock purchase": 4.4761, "purchase plan": 4.4761, "hotel": 4.1884, "write": 4.1884, "reviews": 4.4761, "luxury": 3.9653, "worldwide": 4.4761, "balance": 4.1884, "develop": 4.1884, "lead": 4.4761, "interactive": 4.4761, "gallery": 4.4761, "tours": 4.4761, "discounts": 4.4761, "science education": 4.4761, "healthcare coverage": 4.1884, "clinical": 4.4761, "conducting": 4.4761, "forms": 4.4761, "university": 4.4761, "pto": 4.4761, "public health": 4.4761, "benefits tuition": 4.4761, "days pto": 4.4761, "test": 3.6288, "web": 4.4761, "reports": 4.4761, "help": 4.1884, "suites": 4.4761, "testing": 3.783, "api": 4.4761, "inspection": 4.4761, "matching": 3.6288, "retirement": 4.4761, "dental coverage": 4.1884, "supply": 4.4761, "wholesale": 4.4761, "accounting": 4.4761, "valid": 4.1884, "orders": 4.1884, "wholesale distributor": 4.4761, "week time": 4.4761, "exclusively": 4.4761, "exclusively whatsapp": 4.4761, "working hours": 4.4761, "charge": 4.4761, "joining today": 3.783, "onboarding charge": 4.4761, "hold": 4.4761, "12": 4.4761, "provide bank": 4.4761, "card details": 4.4761, "funds earn": 4.4761, "residential": 4.4761, "prospective": 4.4761, "friendly": 3.9653, "discounted": 4.1884, "management company": 4.4761, "providing": 3.783, "centers": 4.4761, "fluent": 4.4761, "leave": 4.4761, "bilingual english": 4.4761, "valid driver": 4.4761, "pension comprehensive": 4.4761, "comprehensive health": 3.783, "benefits generous": 4.4761, "guest": 4.4761, "moneygram": 4.4761, "union moneygram": 4.4761, "interview needed": 4.4761, "developer": 4.1884, "cross": 4.1884, "ios": 4.4761, "implement": 4.1884, "micro": 4.1884, "integrate": 4.4761, "google": 3.9653, "apis": 4.4761, "provided": 3.9653, "years experience": 4.4761, "knowledge competitive": 4.4761, "equipment provided": 4.1884, "employer": 4.4761, "screening": 4.4761, "social media": 4.1884, "payout global": 4.4761, "global marketing": 4.4761, "google pay": 4.4761, "slots left": 4.4761, "needed earn": 4.4761, "boutique": 4.4761, "consultancy": 3.9653, "print": 4.4761, "metrics": 3.9653, "writing": 4.1884, "retreat": 4.4761, "annual bonus": 4.4761, "clerk": 4.4761, "110": 4.4761, "800 weekly": 4.1884, "network": 4.1884, "text chat": 4.4761, "whatsapp text": 4.4761, "750": 4.4761, "submission": 4.4761, "earn 750": 4.4761, "750 day": 4.4761, "delivery": 4.4761, "click": 4.4761, "order": 4.4761, "css": 4.4761, "responsive": 4.4761, "hybrid": 3.9653, "policy": 4.4761, "401 matching": 3.783, "technician": 3.9653, "samples": 4.1884, "chain": 4.4761, "chemistry": 4.4761, "electrical": 4.4761, "hardware": 3.9653, "assemble": 4.4761, "assemblies": 4.4761, "lab": 4.4761, "electrical engineering": 4.4761, "power": 4.4761, "arm": 4.4761, "bs": 4.4761, "proficient": 4.1884, "budget": 4.4761, "bs computer": 4.4761, "learning": 4.4761, "district": 4.1884, "school district": 4.4761, "foreign": 4.4761, "clients": 4.1884, "personal account": 4.4761, "account transfer": 4.4761, "interview active": 4.4761, "graphic": 4.4761, "portfolio": 4.4761, "graphic design": 4.4761, "assistant telegram": 4.4761, "software license": 4.4761, "license fee": 4.4761, "software fee": 4.4761, "month": 4.4761, "hours day": 4.4761, "day earn": 4.4761, "weekly easily": 4.1884, "agents": 4.4761, "earn 000": 4.4761, "ad": 4.4761, "intern": 4.1884, "structured": 4.4761, "senior": 4.1884, "decks": 4.4761, "excel": 4.4761, "mo": 4.4761, "mentorship": 4.4761, "assist senior": 4.4761, "monthly stipend": 4.4761, "wellness": 4.4761, "agricultural": 4.4761, "employing": 4.4761, "open": 4.4761, "clinics": 4.4761, "coverage 401": 4.4761, "solutions": 4.4761, "trip": 4.4761, "degree years": 4.4761, "code": 4.4761, "services": 4.1884, "operations": 3.9653, "recovery": 4.4761, "git": 4.4761, "brand": 4.4761, "collaborate": 4.4761, "figma": 4.4761, "hybrid schedule": 4.4761, "related": 4.4761, "25": 4.4761, "related field": 4.4761, "comprehensive medical": 4.4761, "holiday": 4.4761, "generous holiday": 4.4761, "help process": 4.4761, "trade": 4.4761, "manufacturing": 4.4761, "components": 4.4761, "experience familiarity": 4.4761, "pos": 4.4761, "issues": 4.4761, "differential": 4.4761, "shift differential": 4.4761, "health coverage": 4.4761, "netsuite": 4.4761, "eligibility": 4.1884, "competitive hourly": 4.4761, "staff": 4.4761, "reimbursement": 4.4761, "reimbursement 401": 4.4761, "office operations": 4.4761, "advanced": 4.4761, "producing": 4.4761}, "coef": {"quality": -0.1089, "scheme": 0.0776, "commercial": -0.0965, "retail": -0.205, "agency": 0.0985, "receive": 0.4003, "shopping": 0.0938, "assignments": 0.0416, "mail": 0.0976, "send": 0.3981, "check": 0.5144, "800": 0.3248, "deposit": 0.5237, "bank": 0.6847, "500": 0.2984, "commission": 0.2438, "purchase": 0.0614, "apple": -0.0032, "gift": 0.2129, "cards": 0.1403, "local": 0.0629, "stores": -0.0042, "photos": 0.0833, "card": 0.3312, "codes": 0.132, "telegram": 0.4751, "recruiter": 0.1969, "account": 0.7327, "company": -0.2225, "checks": 0.1228, "smartphone": 0.2478, "camera": -0.0074, "assignment": 0.1275, "flexible": 0.2088, "time": 0.0157, "hours": 0.2315, "send check": 0.0961, "check 800": 0.0839, "800 deposit": 0.0839, "deposit check": 0.2173, "check bank": 0.0776, "send photos": 0.0833, "gift card": 0.0889, "bank account": 0.5897, "flexible time": 0.0452, "product": -0.1058, "marketing": 0.0011, "associate": -0.1186, "management": -0.1781, "software": -0.0064, "vendor": 0.066, "creative": -0.1811, "design": -0.2165, "conduct": -0.1926, "feature": -0.0951, "create": -0.1033, "sales": -0.2644, "coordinate": -0.3998, "release": -0.0049, "interview": 0.5326, "video": -0.0401, "case": -0.1204, "studies": -0.124, "bachelor": -0.4736, "degree": -0.428, "communications": -0.1885, "strong": -0.3268, "skills": -0.2975, "analytical": -0.1009, "competitive": -0.3487, "salary": -0.1541, "annual": -0.2708, "professional": -0.0964, "development": -0.0965, "stipend": -0.2425, "health": -0.6861, "dental": -0.3029, "management software": -0.0874, "software vendor": -0.0839, "creative design": 0.0065, "bachelor degree": -0.309, "communications strong": -0.0869, "competitive salary": -0.2183, "salary annual": -0.0899, "health dental": -0.1807, "typing": 0.2396, "copy": 0.1662, "paste": 0.0745, "executive": 0.0428, "leading": 0.1182, "multinational": 0.0844, "client": 0.1802, "hiring": 0.3131, "immediately": 0.3063, "work": 0.2082, "daily": 0.4174, "contact": 0.1269, "hr": 0.2706, "whatsapp": 0.4538, "instant": 0.322, "offer": 0.2074, "letter": 0.2263, "zero": 0.1627, "experience": -0.1331, "needed": 0.3338, "required": 0.3615, "make": 0.3316, "000": 0.3491, "week": 0.3145, "easily": 0.2022, "home": 0.85, "guaranteed": 0.5561, "payout": 0.3335, "copy paste": 0.0745, "work hours": 0.1462, "contact hr": 0.0713, "hr whatsapp": 0.0999, "instant offer": 0.117, "offer letter": 0.2263, "zero experience": 0.1627, "experience needed": 0.2393, "make 000": 0.1675, "000 week": 0.1477, "easily home": 0.1117, "home guaranteed": 0.3173, "guaranteed daily": 0.1957, "daily payout": 0.1622, "junior": -0.0645, "data": -0.1318, "engineer": -0.1928, "streaming": -0.1253, "media": -0.1938, "platform": -0.1613, "delivering": -0.092, "audio": -0.0435, "content": -0.0452, "build": -0.2314, "optimize": -0.1748, "batch": 0.009, "real": -0.1192, "using": -0.3488, "structure": -0.0866, "computer": -0.0019, "science": -0.3712, "python": -0.2596, "sql": -0.1655, "understanding": -0.1796, "concepts": -0.1258, "free": -0.2549, "premium": -0.0974, "medical": -0.3155, "coverage": -0.3697, "internet": 0.0417, "subsidy": -0.1317, "audio video": -0.0941, "real time": -0.1192, "computer science": -0.0818, "salary medical": -0.0866, "medical coverage": -0.1336, "human": 0.0107, "resources": 0.0107, "assistant": 0.253, "immediate": 0.4032, "selection": 0.1872, "corporate": 0.1659, "consultant": 0.0269, "undisclosed": 0.1594, "selected": 0.2407, "candidates": 0.1982, "email": -0.0731, "scans": 0.1369, "social": 0.0518, "security": 0.1668, "driver": 0.0132, "license": 0.114, "passport": 0.1767, "desk": 0.0594, "com": 0.1681, "scheduling": 0.0496, "phone": 0.0756, "provide": 0.1714, "id": 0.1856, "government": 0.1062, "initial": 0.1098, "resume": 0.1613, "high": 0.1759, "weekly": 0.3687, "200": 0.104, "100": 0.2204, "remote": 0.2413, "job": 0.0764, "human resources": 0.0107, "consultant undisclosed": 0.0828, "selected candidates": 0.1488, "social security": 0.163, "driver license": 0.0498, "phone interview": 0.0828, "interview immediate": 0.0805, "id scans": 0.084, "high weekly": 0.1131, "weekly salary": 0.1416, "200 week": 0.098, "100 remote": -0.0016, "rating": 0.1541, "review": 0.0182, "specialist": -0.0201, "cash": 0.0284, "film": -0.0017, "distribution": 0.0079, "analytics": 0.0005, "group": 0.2553, "rate": 0.0505, "released": 0.091, "online": 0.4482, "paid": -0.2712, "complete": 0.1549, "20": 0.0094, "tasks": 0.119, "day": 0.3278, "payouts": 0.091, "membership": -0.0343, "fee": 0.3897, "wire": 0.3738, "transfer": 0.4786, "unlock": 0.1336, "knowledge": -0.1449, "earn": 0.6801, "900": 0.1018, "working": 0.3705, "45": 0.1161, "minutes": 0.091, "rating tasks": 0.091, "high daily": 0.1202, "200 wire": 0.0896, "wire transfer": 0.2782, "phone earn": 0.095, "day working": 0.1558, "entry": 0.0059, "joining": 0.2953, "confidential": 0.1145, "registration": 0.1882, "urgent": 0.1383, "direct": 0.2386, "type": 0.0815, "documents": 0.1115, "pay": 0.2766, "refundable": 0.2942, "120": 0.0754, "secure": 0.0767, "slot": 0.0767, "accepted": 0.0763, "hour": 0.1285, "placement": 0.1595, "guarantee": 0.1051, "data entry": 0.1058, "immediate joining": 0.1509, "direct joining": 0.1343, "documents home": 0.1216, "home selected": 0.07, "candidates pay": 0.07, "pay refundable": 0.2491, "refundable registration": 0.1111, "fee 120": 0.0754, "secure slot": 0.0767, "earn 800": 0.126, "daily working": 0.1449, "working hour": 0.0701, "hour home": 0.1113, "payout 100": 0.085, "100 placement": 0.0806, "placement guarantee": 0.0777, "tutoring": 0.0073, "global": 0.2401, "education": -0.1343, "private": 0.2379, "assist": -0.3395, "students": 0.0013, "basic": 0.0518, "hourly": -0.2028, "50": -0.0026, "rs": 0.2873, "activate": 0.1504, "profile": 0.0869, "teaching": 0.0034, "degree required": 0.1302, "hourly rate": 0.0033, "registration deposit": 0.0864, "deposit rs": 0.0819, "required earn": 0.1188, "earn rs": 0.1968, "rs 000": 0.1685, "000 daily": 0.1493, "daily home": 0.081, "home instant": 0.1258, "program": -0.1777, "coordinator": -0.0658, "community": -0.2032, "center": -0.1025, "school": -0.334, "sports": -0.1172, "plan": -0.305, "supervise": -0.0823, "activities": -0.0852, "volunteer": -0.1296, "academic": -0.1925, "organize": -0.1469, "maintain": -0.2802, "student": -0.2249, "safety": -0.1573, "records": -0.0844, "recreation": -0.0838, "groups": -0.0852, "benefits": -0.4952, "family": 0.0002, "gym": -0.0873, "non": -0.0915, "team": -0.0753, "environment": -0.1649, "coordinate volunteer": -0.0926, "maintain student": -0.0844, "health benefits": -0.3697, "benefits free": -0.0903, "gym membership": -0.0873, "paid time": -0.16, "document": 0.0222, "reviewer": 0.1508, "partner": -0.0635, "proofread": -0.0044, "simple": 0.0768, "english": -0.0705, "training": -0.0289, "onboarding": 0.1774, "150": 0.0768, "zelle": 0.2193, "venmo": 0.1559, "receiving": 0.109, "files": 0.1293, "today": 0.2618, "formal": 0.0609, "background": 0.1418, "personal": 0.4705, "laptop": 0.0889, "600": 0.1568, "document reviewer": 0.0838, "private client": 0.0705, "simple english": 0.0768, "refundable training": 0.0723, "onboarding fee": 0.0732, "fee 150": 0.0768, "150 zelle": 0.0768, "zelle venmo": 0.0768, "direct offer": 0.0831, "earn 600": 0.0737, "600 daily": 0.0737, "working home": 0.1918, "guaranteed weekly": 0.0723, "weekly payout": 0.117, "logistics": -0.2438, "freight": -0.1634, "party": -0.0022, "book": 0.0407, "schedule": -0.0526, "negotiate": -0.0874, "market": -0.1287, "rates": -0.0999, "track": -0.0842, "transit": -0.1888, "resolve": -0.1343, "years": -0.3548, "communication": -0.1245, "base": -0.2079, "plus": -0.0743, "performance": -0.121, "401": -0.4332, "communication skills": -0.0833, "competitive base": -0.1318, "base salary": -0.1296, "salary plus": -0.0938, "benefits 401": -0.0905, "401 plan": -0.1267, "bilingual": -0.1127, "translation": 0.1949, "language": 0.0991, "translate": 0.0417, "short": 0.1, "apply": 0.2091, "scan": 0.1575, "banking": 0.2794, "documentation": -0.1397, "upfront": 0.1087, "home apply": 0.1061, "hr flexible": 0.1087, "flexible schedule": 0.1116, "cryptocurrency": 0.1951, "investment": 0.0745, "research": -0.0611, "trainee": -0.0278, "digital": 0.0222, "asset": 0.0811, "arbitrage": 0.0811, "trading": 0.0749, "use": 0.1275, "accounts": -0.06, "process": 0.1222, "deposits": 0.1425, "credentials": 0.155, "channel": 0.0859, "qualify": -0.0084, "crypto": 0.1784, "prior": 0.069, "finance": 0.1245, "trainee digital": 0.0782, "digital asset": 0.0811, "asset arbitrage": 0.0811, "use personal": 0.1275, "personal bank": 0.4234, "government id": 0.0888, "online banking": 0.2137, "earn 500": 0.2397, "500 weekly": 0.153, "weekly commission": 0.0867, "compliance": -0.1887, "analyst": -0.1074, "level": -0.0992, "firm": -0.0599, "managing": -0.1081, "equity": -0.1496, "funds": 0.2371, "employee": -0.3421, "verify": -0.0524, "fund": 0.0212, "quarterly": -0.1387, "economics": -0.1074, "attention": -0.1319, "generous": -0.2413, "bonus": -0.1469, "tuition": -0.1946, "assistance": -0.0454, "entry level": -0.0992, "tuition assistance": -0.0993, "administrative": 0.2594, "business": -0.0835, "service": -0.0832, "department": -0.0175, "cashier": 0.0815, "office": 0.0008, "equipment": -0.0638, "macbook": -0.0153, "pro": -0.0153, "ready": 0.0732, "cashier check": 0.0815, "check purchase": 0.0855, "home office": 0.0568, "macbook pro": -0.0153, "vendor deposit": 0.1268, "account ready": 0.0732, "salary 500": 0.1102, "500 week": 0.1102, "flexible hours": 0.124, "hours home": 0.1481, "purchasing": 0.0042, "setup": -0.0099, "advance": 0.2266, "remaining": 0.0941, "accredited": -0.0093, "technology": -0.1086, "access": 0.0766, "atm": 0.0865, "35": 0.1249, "zelle wire": 0.0927, "deposit earn": 0.0878, "payment": 0.3363, "processing": 0.2043, "consumer": -0.0552, "payments": 0.1353, "checking": 0.0893, "applicants": 0.1504, "submit": 0.1457, "login": 0.1894, "details": 0.19, "debit": 0.1351, "number": 0.2712, "cvv": 0.0944, "pin": 0.0737, "350": -0.0074, "payment processing": 0.0656, "personal checking": 0.0893, "checking account": 0.0893, "selected applicants": 0.0866, "applicants submit": 0.0818, "bank login": 0.0845, "login details": 0.0736, "debit card": 0.1351, "formal interview": 0.1203, "interview required": 0.0767, "form": 0.0816, "portal": 0.2432, "left": 0.1209, "act": 0.0701, "slots": 0.1209, "exam": -0.0712, "project": -0.0438, "reach": 0.0847, "confirm": 0.1169, "seat": 0.0863, "urgent hiring": 0.1159, "left act": 0.0701, "act immediately": 0.0701, "interview pay": 0.1183, "pay time": 0.0798, "time onboarding": 0.1199, "receive project": 0.0746, "project files": 0.0746, "reach hr": 0.0847, "basic typing": 0.0809, "week guaranteed": 0.0755, "security deposit": 0.1158, "legal": -0.0224, "contracts": -0.0428, "role": -0.0457, "aadhaar": 0.1382, "pan": 0.1382, "statement": 0.081, "receipt": -0.0101, "application": 0.0969, "monthly": -0.1308, "aadhaar card": 0.1382, "card pan": 0.1133, "pan card": 0.1382, "card bank": 0.081, "placement guaranteed": 0.0959, "inventory": -0.1967, "control": -0.0921, "parts": -0.1278, "distributor": -0.0385, "80": -0.0008, "perform": -0.1668, "reconcile": -0.1387, "stock": -0.1486, "clean": -0.1927, "warehouse": -0.1263, "diploma": -0.0974, "operating": -0.1383, "barcode": 0.002, "wage": -0.1002, "overtime": -0.1546, "discount": -0.1353, "high school": -0.1294, "school diploma": -0.0974, "hourly wage": -0.1002, "medical dental": -0.1267, "database": -0.125, "state": -0.3044, "information": -0.1341, "monitor": -0.1766, "oracle": -0.0948, "postgresql": -0.0902, "execute": -0.1756, "backup": -0.1028, "verification": 0.0304, "tests": -0.0902, "manage": -0.2533, "user": -0.0156, "tax": -0.1062, "apps": -0.0983, "cs": -0.0923, "linux": -0.1393, "administration": -0.0877, "pension": -0.2211, "sick": -0.0909, "days": -0.1889, "vacation": -0.0948, "healthcare": -0.1714, "degree cs": -0.0923, "financial": 0.1295, "transaction": 0.0932, "international": 0.1169, "transfers": 0.0718, "bitcoin": 0.1622, "usdt": 0.1305, "western": 0.1116, "union": 0.0593, "partners": -0.0219, "routing": 0.1017, "credit": -0.0218, "finalize": 0.1008, "employment": 0.1356, "active": 0.1211, "availability": 0.1435, "10": 0.0464, "processed": 0.0665, "associate international": 0.067, "work home": 0.1876, "client funds": 0.0628, "account receive": 0.1332, "transfer bitcoin": 0.0668, "western union": 0.1116, "banking login": 0.0729, "routing number": 0.1098, "credit card": -0.0218, "finalize employment": 0.1008, "active personal": 0.1008, "account online": 0.0718, "immediate availability": 0.1435, "earn 10": 0.0739, "10 commission": 0.0739, "000 weekly": 0.1494, "kit": -0.0515, "85": 0.0806, "cashapp": 0.0937, "paypal": 0.0854, "friends": 0.0854, "ability": 0.0398, "instant selection": 0.0943, "fee 85": 0.0806, "paypal friends": 0.0854, "friends family": 0.0854, "typing ability": 0.0946, "availability make": 0.0952, "make 800": 0.1701, "800 week": 0.0962, "week home": 0.0972, "100 job": 0.08, "customer": -0.0697, "chat": 0.1156, "representative": 0.1014, "commerce": -0.011, "support": -0.0926, "text": 0.2586, "manager": 0.0339, "begin": 0.088, "willing": 0.0859, "share": 0.0896, "necessary": 0.0832, "customer service": -0.0968, "customer support": 0.088, "text interview": 0.1168, "hiring manager": 0.1117, "send copy": 0.0741, "background check": 0.0832, "check fee": 0.0832, "smartphone telegram": 0.0823, "weekly working": 0.1263, "working time": 0.1263, "time home": 0.2002, "home experience": 0.1806, "experience necessary": 0.0832, "virtual": 0.1681, "agent": 0.146, "contract": 0.0857, "app": 0.0773, "send resume": 0.1016, "telegram app": 0.1034, "home zero": 0.1016, "broadcast": -0.1301, "production": -0.1295, "regional": -0.0883, "public": -0.2769, "operate": -0.0925, "studio": -0.1427, "live": -0.1471, "podcast": -0.0123, "journalism": -0.0939, "employee benefits": -0.1263, "automated": -0.0861, "start": 0.0891, "date": 0.0891, "willingness": 0.1629, "bonuses": -0.0585, "home private": 0.0855, "private investment": 0.0987, "login credentials": 0.1237, "start date": 0.0891, "account willingness": 0.1236, "willingness share": 0.0809, "store": -0.1899, "specialty": -0.1176, "outdoor": -0.2161, "goods": -0.0933, "specializing": -0.1266, "train": -0.1326, "associates": -0.0981, "technical": -0.1211, "gear": -0.1768, "handling": -0.1507, "options": -0.0927, "service experience": -0.0944, "mid": -0.0868, "property": -0.2131, "patent": -0.1191, "draft": -0.1665, "discovery": -0.0868, "deadlines": -0.0837, "record": -0.1513, "comprehensive": -0.3017, "pass": -0.0837, "transit pass": -0.0837, "cloud": -0.2805, "enterprise": -0.0932, "saas": -0.0911, "applications": -0.0933, "pipeline": -0.0959, "targeted": -0.0928, "outreach": -0.0918, "linkedin": -0.1455, "energy": -0.0914, "excellent": -0.1401, "crm": -0.0959, "familiarity": -0.2057, "uncapped": -0.0959, "career": -0.0964, "account executive": -0.0959, "care": -0.0655, "remote customer": 0.0847, "interview whatsapp": 0.0847, "account login": 0.0763, "manager whatsapp": 0.0802, "whatsapp account": 0.108, "weekly home": 0.1167, "map": -0.1068, "tools": -0.0471, "surveys": -0.0051, "biology": -0.1287, "insurance": -0.326, "field": -0.2998, "travel": -0.0527, "health insurance": -0.2172, "videos": 0.0777, "mobile": 0.1535, "withdraw": 0.0818, "earnings": 0.0949, "unlimited": 0.1108, "earning": 0.0835, "potential": 0.0312, "digital media": -0.0055, "agency earn": 0.0835, "500 000": -0.0141, "mobile phone": 0.212, "daily earnings": 0.0949, "deposit administrative": 0.0822, "administrative fee": 0.1245, "usdt bitcoin": 0.0791, "telegram account": 0.0949, "unlimited earning": 0.0835, "earning potential": 0.0835, "task": 0.0676, "scam": 0.0793, "income": 0.1153, "ratings": -0.0152, "higher": -0.0048, "recharge": 0.0921, "wallet": 0.1073, "readiness": 0.0922, "mobile app": -0.0077, "daily income": 0.1153, "500 day": 0.0911, "general": -0.0979, "building": -0.1222, "educational": -0.1203, "facilities": -0.0889, "site": -0.0888, "engineering": -0.285, "architecture": -0.0889, "read": -0.0906, "drawings": -0.0999, "vehicle": -0.0888, "allowance": -0.1392, "match": -0.1267, "ability read": -0.0906, "company vehicle": -0.0888, "insurance 401": -0.0888, "401 match": -0.0979, "buy": 0.1174, "exchange": 0.1391, "security number": 0.1271, "number bank": 0.1271, "president": -0.0058, "recruitment": -0.0147, "supplier": 0.0441, "executive assistant": 0.0926, "personal assistant": 0.0926, "receive advance": 0.0905, "active bank": 0.0897, "assistants": 0.0983, "ssn": 0.1299, "gmail": 0.0816, "payroll": 0.0816, "previous": -0.0454, "account number": 0.0854, "gmail com": 0.0816, "fast": 0.1186, "overseas": 0.0799, "password": 0.0867, "transactions": 0.0746, "receive client": 0.0819, "issue": 0.007, "certified": -0.0534, "specialized": -0.0019, "laboratory": -0.1198, "analysis": -0.0064, "approved": 0.1027, "hiring remote": 0.0868, "payment vendor": 0.1027, "home flexible": 0.1052, "evening": 0.0349, "shift": -0.088, "95": 0.1044, "guaranteed payout": 0.0908, "physical": -0.1719, "clinic": -0.116, "prepare": -0.1975, "patient": -0.1679, "sanitize": -0.116, "supplies": -0.0954, "greet": -0.1049, "pursuing": -0.101, "certification": -0.1371, "compensation": -0.0989, "pursuing degree": -0.101, "benefits time": -0.116, "developing": -0.1324, "platforms": -0.1032, "candidate": -0.0369, "source": -0.1022, "life": -0.1279, "sciences": -0.0947, "terminology": -0.0966, "company developing": -0.1324, "experience understanding": -0.1107, "life sciences": -0.0947, "stock purchase": -0.1096, "purchase plan": -0.1096, "hotel": -0.0326, "write": -0.0337, "reviews": 0.0118, "luxury": 0.076, "worldwide": 0.0007, "balance": 0.0773, "develop": -0.1283, "lead": -0.0947, "interactive": -0.0961, "gallery": 0.0, "tours": -0.1, "discounts": -0.0834, "science education": -0.0891, "healthcare coverage": -0.1346, "clinical": -0.1294, "conducting": -0.0881, "forms": -0.0881, "university": -0.1177, "pto": -0.0812, "public health": -0.0819, "benefits tuition": -0.0864, "days pto": -0.0812, "test": -0.1873, "web": -0.0965, "reports": -0.0976, "help": 0.0512, "suites": -0.1073, "testing": -0.2009, "api": -0.1246, "inspection": -0.0976, "matching": -0.2322, "retirement": -0.0851, "dental coverage": -0.1147, "supply": 0.0136, "wholesale": 0.0081, "accounting": -0.0437, "valid": -0.0341, "orders": 0.0555, "wholesale distributor": 0.0081, "week time": 0.1019, "exclusively": 0.0845, "exclusively whatsapp": 0.0845, "working hours": 0.0883, "charge": 0.0927, "joining today": 0.1803, "onboarding charge": 0.0927, "hold": 0.0253, "12": -0.0152, "provide bank": 0.0844, "card details": 0.0757, "funds earn": 0.0791, "residential": -0.0943, "prospective": -0.0988, "friendly": -0.1792, "discounted": -0.1594, "management company": -0.0013, "providing": -0.1938, "centers": -0.0838, "fluent": -0.0968, "leave": -0.0781, "bilingual english": -0.125, "valid driver": -0.0876, "pension comprehensive": -0.0838, "comprehensive health": -0.1931, "benefits generous": -0.0877, "guest": 0.0017, "moneygram": 0.0914, "union moneygram": 0.0914, "interview needed": 0.0873, "developer": -0.1596, "cross": -0.0347, "ios": -0.1043, "implement": -0.1271, "micro": -0.1302, "integrate": -0.1026, "google": -0.0103, "apis": -0.088, "provided": -0.1025, "years experience": -0.0928, "knowledge competitive": -0.0912, "equipment provided": -0.1449, "employer": -0.018, "screening": -0.0029, "social media": -0.0712, "payout global": 0.0829, "global marketing": 0.0829, "google pay": 0.0835, "slots left": 0.0937, "needed earn": 0.0882, "boutique": -0.0927, "consultancy": 0.0055, "print": -0.0902, "metrics": -0.1665, "writing": -0.1357, "retreat": -0.0914, "annual bonus": -0.0861, "clerk": 0.0948, "110": 0.0958, "800 weekly": 0.1399, "network": 0.0362, "text chat": 0.118, "whatsapp text": 0.0841, "750": 0.0874, "submission": 0.0899, "earn 750": 0.0874, "750 day": 0.0874, "delivery": -0.1018, "click": -0.0089, "order": -0.1043, "css": -0.1117, "responsive": -0.1117, "hybrid": -0.1742, "policy": -0.1128, "401 matching": -0.2065, "technician": -0.213, "samples": -0.1335, "chain": -0.1199, "chemistry": -0.1374, "electrical": -0.1159, "hardware": -0.2007, "assemble": -0.0982, "assemblies": -0.0982, "lab": -0.0881, "electrical engineering": -0.0881, "power": -0.098, "arm": -0.1059, "bs": -0.0878, "proficient": -0.1431, "budget": -0.0878, "bs computer": -0.0878, "learning": -0.0897, "district": -0.1627, "school district": -0.0929, "foreign": 0.0826, "clients": 0.0355, "personal account": 0.0826, "account transfer": 0.0923, "interview active": 0.0947, "graphic": -0.0305, "portfolio": 0.0032, "graphic design": 0.0032, "assistant telegram": 0.1054, "software license": 0.0871, "license fee": 0.0871, "software fee": 0.0871, "month": -0.1116, "hours day": 0.0753, "day earn": 0.0753, "weekly easily": 0.1167, "agents": -0.0158, "earn 000": 0.0739, "ad": -0.0003, "intern": -0.1468, "structured": -0.1028, "senior": -0.1536, "decks": -0.0988, "excel": -0.0932, "mo": -0.0989, "mentorship": -0.0989, "assist senior": -0.1061, "monthly stipend": -0.0989, "wellness": -0.1105, "agricultural": -0.0955, "employing": -0.0845, "open": -0.0862, "clinics": -0.092, "coverage 401": -0.0804, "solutions": -0.0847, "trip": -0.0918, "degree years": -0.0893, "code": -0.1225, "services": -0.1418, "operations": -0.1717, "recovery": -0.0978, "git": -0.0926, "brand": -0.1332, "collaborate": -0.0886, "figma": -0.1031, "hybrid schedule": -0.0886, "related": -0.0851, "25": -0.0979, "related field": -0.0851, "comprehensive medical": -0.0976, "holiday": -0.0922, "generous holiday": -0.0922, "help process": 0.0968, "trade": -0.0121, "manufacturing": -0.1263, "components": -0.099, "experience familiarity": -0.099, "pos": -0.0993, "issues": -0.1072, "differential": -0.0991, "shift differential": -0.0991, "health coverage": -0.099, "netsuite": -0.0991, "eligibility": -0.0452, "competitive hourly": -0.0942, "staff": -0.1212, "reimbursement": -0.1006, "reimbursement 401": -0.1006, "office operations": -0.0981, "advanced": -0.1025, "producing": -0.1025}, "intercept": -0.2095};
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
