import streamlit as st
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import googletrans
from googletrans import Translator
from urllib.parse import urlparse

# === Dataset utilities ===
def _normalize_and_select(df: pd.DataFrame) -> pd.DataFrame:
    # Try to find label and text columns
    candidates_label = ["label", "target", "category", "tag", "class"]
    candidates_text = ["text", "message", "sms", "content", "msg"]
    label_col = None
    text_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if label_col is None and cl in candidates_label:
            label_col = c
        if text_col is None and cl in candidates_text:
            text_col = c
    # fallback to 1st and 2nd columns if not found
    if label_col is None or text_col is None:
        cols = list(df.columns)
        if len(cols) >= 2:
            label_col = label_col or cols[0]
            text_col = text_col or cols[1]
        else:
            raise ValueError("Dataset must have at least two columns (label and text)")

    out = df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"}).copy()
    out["label"] = out["label"].astype(str).str.strip().str.lower().replace({
        "spam": "spam", "1": "spam", "yes": "spam", "true": "spam", "junk": "spam",
        "ham": "ham", "0": "ham", "no": "ham", "false": "ham", "normal": "ham"
    })
    out = out[out["label"].isin(["spam", "ham"])].dropna()
    out["label_num"] = out.label.map({"ham": 0, "spam": 1})
    return out

def _read_csv_any(path: str) -> pd.DataFrame:
    # Try common encodings
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # Fallback default
    return pd.read_csv(path)

def load_combined_datasets() -> pd.DataFrame:
    datasets = []
    # Base English dataset
    base_path = os.path.join("dataset", "spam.csv")
    if os.path.exists(base_path):
        df = _read_csv_any(base_path)
        # Original kaggle spam.csv has columns v1, v2
        if set(["v1", "v2"]).issubset(set(df.columns)):
            df = df[["v1", "v2"]]
            df.columns = ["label", "text"]
            df["label_num"] = df.label.map({"ham": 0, "spam": 1})
            datasets.append(df)
        else:
            datasets.append(_normalize_and_select(df))

    # Bengali dataset
    bn_path = os.path.join("dataset", "banglaspam2.csv")
    if os.path.exists(bn_path):
        df_bn = _read_csv_any(bn_path)
        datasets.append(_normalize_and_select(df_bn))

    # Hindi dataset
    hi_path = os.path.join("dataset", "spam_hindi.csv")
    if os.path.exists(hi_path):
        df_hi = _read_csv_any(hi_path)
        datasets.append(_normalize_and_select(df_hi))

    if not datasets:
        raise FileNotFoundError("No datasets found. Expected at least dataset/spam.csv")

    combined = pd.concat(datasets, ignore_index=True)
    combined.dropna(subset=["text", "label_num"], inplace=True)
    return combined

# === Load or train model (fast path, multilingual via combined datasets, no augmentation) ===
@st.cache_resource
def load_model():
    df = load_combined_datasets()

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", MultinomialNB(alpha=0.5))
    ])
    model.fit(df["text"], df["label_num"])
    return model

# === Optional: multilingual augmented training (slow) ===
@st.cache_resource
def load_model_multilingual():
    # Use combined multilingual datasets as base; optionally augment further
    df = load_combined_datasets()

    def translate_series(texts, dest_lang):
        out = []
        for t in texts:
            try:
                out.append(Translator().translate(str(t), dest=dest_lang).text)
            except Exception:
                out.append(str(t))
        return out

    df_sample = df.sample(frac=1.0, random_state=42)
    max_aug = min(len(df_sample), 600)  # reduced to speed up
    df_aug_base = df_sample.head(max_aug).copy()

    try:
        hi_texts = translate_series(df_aug_base["text"].tolist(), "hi")
        bn_texts = translate_series(df_aug_base["text"].tolist(), "bn")
        df_hi = pd.DataFrame({
            "label": df_aug_base["label"].values,
            "text": hi_texts,
            "label_num": df_aug_base["label_num"].values
        })
        df_bn = pd.DataFrame({
            "label": df_aug_base["label"].values,
            "text": bn_texts,
            "label_num": df_aug_base["label_num"].values
        })
        df_train = pd.concat([df, df_hi, df_bn], ignore_index=True)
    except Exception:
        df_train = df

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", MultinomialNB(alpha=0.5))
    ])
    model.fit(df_train["text"], df_train["label_num"])
    return model

model = load_model()
translator = Translator()

# === Helper: Translate text to English (always translate to normalize mixed languages) ===
def translate_to_english(text):
    try:
        # Always translate to English to avoid partial English being misdetected as fully English
        return translator.translate(text, dest="en").text
    except Exception:
        return text

# === Helper: Ensure spam if any-language translation looks like spam ===
def is_spam_by_rules_multi(text: str) -> bool:
    # Check original
    if is_obvious_spam(str(text)):
        return True
    # Check English translation
    if is_obvious_spam(translate_to_english(str(text))):
        return True
    # Check Hindi/Bengali translations as well
    try:
        hi = translator.translate(str(text), dest="hi").text
        if is_obvious_spam(hi):
            return True
    except Exception:
        pass
    try:
        bn = translator.translate(str(text), dest="bn").text
        if is_obvious_spam(bn):
            return True
    except Exception:
        pass
    return False

# === Helper: Phishing detection ===
def has_phishing_url(text):
    # detect URLs with or without explicit scheme
    urls = re.findall(r'(https?://\S+|www\.\S+|\b[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/\S*)?)', text)
    # include English + Hindi + Bengali phishing-related keywords
    keywords = [
        # English
        "login", "verify", "update", "account", "secure", "bank",
        # Hindi
        "लॉगिन", "सत्यापित", "वेरिफाई", "अपडेट", "खाता", "सिक्योर", "बैंक",
        # Bengali
        "লগইন", "ভেরিফাই", "আপডেট", "অ্যাকাউন্ট", "সিকিউর", "ব্যাংক"
    ]
    for url in urls:
        parsed = urlparse(url)
        for word in keywords:
            if word in parsed.netloc or word in parsed.path:
                return True
    return False

# === Helper: Simple keyword rule to catch obvious spam phrases ===
def is_obvious_spam(text_en: str) -> bool:
    text_l = text_en.lower()
    # English + Hindi + Bengali spam keywords (common variants/transliterations included)
    keywords = [
        # English
        "congratulations", "congrats", "won", "winner", "gift card", "giftcard",
        "amazon", "prize", "reward", "claim", "click here", "limited time",
        "urgent", "free", "action required", "verify your account", "bank account",
        "government", "irs", "refund", "delivery updates", "order", "credit score",
        "credit card", "job offer", "payment information", "suspicious activity",
        "2fa", "two-factor", "your boss", "transfer", "overpayment", "group text",
        "selected to receive", "claim your reward", "act fast", "limited offer",
        "thank you for paying", "loyalty gift",
        # Currency symbols
        "$", "₹", "€", "৳",
        # Hindi
        "बधाई", "विजेता", "जीता", "जीत", "उपहार", "गिफ्ट", "कार्ड", "इनाम",
        "पुरस्कार", "रिवॉर्ड", "दावा", "क्लिक करें", "यहां क्लिक करें", "अभी",
        "सीमित समय", "फ्री", "मुफ्त", "तुरंत कार्य", "खाता सत्यापित", "सरकार",
        "रिफंड", "डिलीवरी अपडेट", "ऑर्डर", "क्रेडिट स्कोर", "क्रेडिट कार्ड",
        "नौकरी का ऑफर", "भुगतान जानकारी", "संदिग्ध गतिविधि", "टू फैक्टर",
        "बॉस", "ट्रांसफर", "ओवरपेमेंट", "समूह संदेश", "इनाम का दावा",
        # Bengali
        "অভিনন্দন", "কনগ্রাচস", "জিতেছেন", "জয়ী", "বিজয়ী", "উপহার",
        "গিফট", "কার্ড", "পুরস্কার", "রিওয়ার্ড", "রিওয়ার্ড", "ক্লেইম",
        "এখানে ক্লিক", "এখনই", "সীমিত সময়", "ফ্রি", "অ্যাকশন প্রয়োজন",
        "অ্যাকাউন্ট যাচাই", "সরকার", "রিফান্ড", "ডেলিভারি আপডেট", "অর্ডার",
        "ক্রেডিট স্কোর", "ক্রেডিট কার্ড", "চাকরির অফার", "পেমেন্ট তথ্য",
        "সন্দেহজনক কার্যকলাপ", "টু ফ্যাক্টর", "বস", "ট্রান্সফার", "ওভারপেমেন্ট",
        "গ্রুপ টেক্সট", "রিওয়ার্ড দাবি"
    ]
    hits = sum(1 for k in keywords if k in text_l)
    # English patterns
    won_combo_en = bool(re.search(r"\bwon\b.*\b(gift|prize|reward|card)\b", text_l))
    claim_here_en = bool(re.search(r"\bclaim\b.*\b(here|now|reward)\b", text_l))
    bank_verify_en = bool(re.search(r"(action required|verify).*(account|bank|payment)", text_l))
    refund_en = bool(re.search(r"\brefund\b.*\b(click|link|claim)\b", text_l))
    delivery_en = bool(re.search(r"(delivery|order).*(update|track)\b", text_l))
    suspicious_en = bool(re.search(r"suspicious activity|2fa|two-factor", text_l))
    # Hindi patterns (e.g., जीता/विजेता + इनाम/उपहार/रिवॉर्ड)
    won_combo_hi = bool(re.search(r"(जीत|जीता|विजेता).*(इनाम|उपहार|रिवॉर्ड|पुरस्कार|कार्ड)", text_l))
    claim_here_hi = bool(re.search(r"(दावा|क्लेम).*(अभी|अब|यहाँ|यहां)", text_l))
    bank_verify_hi = bool(re.search(r"(कार्य आवश्यक|सत्यापित|वेरिफाई).*(खाता|भुगतान)", text_l))
    refund_hi = bool(re.search(r"रिफंड.*(क्लिक|लिंक|दावा)", text_l))
    delivery_hi = bool(re.search(r"(डिलीवरी|आर्डर|ऑर्डर).*(अपडेट|ट्रैक)", text_l))
    suspicious_hi = bool(re.search(r"संदिग्ध गतिविधि|टू फैक्टर", text_l))
    # Bengali patterns (e.g., জিতেছেন/বিজয়ী + পুরস্কার/উপহার/রিওয়ার্ড)
    won_combo_bn = bool(re.search(r"(জিতেছেন|বিজয়ী|জয়ী).*(পুরস্কার|উপহার|রিওয়ার্ড|কার্ড)", text_l))
    claim_here_bn = bool(re.search(r"(ক্লেইম|দাবি).*(এখনই|এখানে)", text_l))
    bank_verify_bn = bool(re.search(r"(অ্যাকশন|যাচাই|ভেরিফাই).*(অ্যাকাউন্ট|পেমেন্ট)", text_l))
    refund_bn = bool(re.search(r"রিফান্ড.*(ক্লিক|লিংক|দাবি)", text_l))
    delivery_bn = bool(re.search(r"(ডেলিভারি|অর্ডার).*(আপডেট|ট্র্যাক)", text_l))
    suspicious_bn = bool(re.search(r"সন্দেহজনক কার্যকলাপ|টু ফ্যাক্টর", text_l))
    return (
        hits >= 2
        or won_combo_en or claim_here_en or bank_verify_en or refund_en or delivery_en or suspicious_en
        or won_combo_hi or claim_here_hi or bank_verify_hi or refund_hi or delivery_hi or suspicious_hi
        or won_combo_bn or claim_here_bn or bank_verify_bn or refund_bn or delivery_bn or suspicious_bn
    )

# === Streamlit UI ===
if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "Dark"

# Sidebar controls (kept functionality intact)
with st.sidebar:
    st.header("⚙️ Settings")
    theme_choice = st.selectbox("Appearance", ["Dark", "Light"], index=0 if st.session_state.ui_theme == "Dark" else 1)
    st.session_state.ui_theme = theme_choice

# Inject theme-specific CSS (no JS; pure CSS so functionality remains untouched)
def _inject_css(theme: str) -> None:
    if theme == "Dark":
        st.markdown(
            """
            <style>
            .stApp { background: #0f172a; }
            .app-card { max-width: 720px; margin: 24px auto; padding: 24px; background: #0b1220; border: 1px solid #1f2a44; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,.35); }
            .app-title { color: #e2e8f0; font-size: 40px; font-weight: 700; text-align: center; margin: 0 0 6px; }
            .app-sub { color: #9aa7bd; text-align: center; margin-bottom: 24px; }
            .stMain .block-container label p { color: #cbd5e1 !important; font-weight: 500; }
            .stMain .block-container textarea { background: #1e293b !important; color: #e2e8f0 !important; border-radius: 12px !important; border: 1px solid #334155 !important; }
            .stMain .block-container textarea::placeholder { color: #94a3b8 !important; }
            .stMain .block-container .stButton>button { width: 100%; background: #2457ff; color: #fff; border: 1px solid #1e40af; border-radius: 28px; padding: 14px 18px; font-weight: 700; transition: transform .06s ease, box-shadow .2s ease, background .2s ease; box-shadow: 0 6px 20px rgba(36,87,255,.35); }
            .stMain .block-container .stButton>button:hover { background: #2a63ff; transform: translateY(-1px); box-shadow: 0 10px 24px rgba(36,87,255,.45); }
            .stMain .block-container .stButton>button:active { transform: translateY(0); }
            .stAlert, .stSuccess, .stWarning, .stError { border-radius: 12px !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp { background: #f6f8fb; }
            .app-card { max-width: 720px; margin: 24px auto; padding: 24px; background: #ffffff; border: 1px solid #e6eaf2; border-radius: 16px; box-shadow: 0 8px 26px rgba(16,24,40,.06); }
            .app-title { color: #0f172a; font-size: 40px; font-weight: 800; text-align: center; margin: 0 0 6px; }
            .app-sub { color: #475569; text-align: center; margin-bottom: 24px; }
            /* Limit color overrides strictly to MAIN content, not sidebar */
            .stMain .block-container, .stMain .block-container * { color: #0f172a; }
            .stMain .block-container .stMarkdown code { background: #eef2f7; color: #0f172a; }
            .stMain .block-container label p { color: #0f172a !important; font-weight: 600; }
            .stMain .block-container textarea { background: #ffffff !important; color: #0f172a !important; border-radius: 12px !important; border: 1px solid #cbd5e1 !important; }
            .stMain .block-container textarea::placeholder { color: #94a3b8 !important; }
            .stMain .block-container .stButton>button { width: 100%; background: #2457ff; color: #fff; border: 1px solid #1e40af; border-radius: 28px; padding: 14px 18px; font-weight: 700; transition: transform .06s ease, box-shadow .2s ease, background .2s ease; box-shadow: 0 6px 20px rgba(36,87,255,.30); }
            .stMain .block-container .stButton>button:hover { background: #2a63ff; transform: translateY(-1px); box-shadow: 0 10px 22px rgba(36,87,255,.40); }
            .stMain .block-container .stButton>button:active { transform: translateY(0); }
            .stAlert, .stSuccess, .stWarning, .stError { border-radius: 12px !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

_inject_css(st.session_state.ui_theme)

# Card header like the provided UI
st.markdown(
    """
    <div class="app-card">
        <h1 class="app-title">Spam Guard</h1>
        <div class="app-sub">Paste your email content below to check for spam.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


 

# === Single Message Classification ===
# Place widgets inside the same card visual container (CSS handles the look)
with st.container():
    message = st.text_area("", placeholder="Enter or paste email text here...", height=260)

if st.button("Check for Spam"):
    if not message.strip():
        st.warning("Please type something.")
    else:
        translated = translate_to_english(message)
        proba = model.predict_proba([translated])[0]
        pred = model.predict([translated])[0]
        # Lift to spam if any-language rule triggers (original/EN/HI/BN)
        if is_spam_by_rules_multi(message):
            pred = 1
        label = "Spam" if pred == 1 else "Ham"
        conf = round(proba[pred] * 100, 2)

        st.markdown(f"🧠 Prediction: `{label}`")
        st.markdown(f"Confidence Score: {conf}%")
        if has_phishing_url(translated):
            st.error("⚠️ Phishing link detected!")

 
