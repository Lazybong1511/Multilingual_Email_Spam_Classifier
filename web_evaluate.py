import os
import re
import time
from typing import List, Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from googletrans import Translator


# === Reuse: minimal copy of dataset/model loader (kept separate from Streamlit UI) ===
def _normalize_and_select(df: pd.DataFrame) -> pd.DataFrame:
    candidates_label = ["label", "target", "category", "tag", "class"]
    candidates_text = ["text", "message", "sms", "content", "msg"]
    label_col, text_col = None, None
    for c in df.columns:
        cl = str(c).strip().lower()
        if label_col is None and cl in candidates_label:
            label_col = c
        if text_col is None and cl in candidates_text:
            text_col = c
    cols = list(df.columns)
    if label_col is None or text_col is None:
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
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def load_combined_datasets() -> pd.DataFrame:
    datasets = []
    base_path = os.path.join("dataset", "spam.csv")
    if os.path.exists(base_path):
        df = _read_csv_any(base_path)
        if set(["v1", "v2"]).issubset(set(df.columns)):
            df = df[["v1", "v2"]]
            df.columns = ["label", "text"]
            df["label_num"] = df.label.map({"ham": 0, "spam": 1})
            datasets.append(df)
        else:
            datasets.append(_normalize_and_select(df))

    bn_path = os.path.join("dataset", "banglaspam2.csv")
    if os.path.exists(bn_path):
        df_bn = _read_csv_any(bn_path)
        datasets.append(_normalize_and_select(df_bn))

    hi_path = os.path.join("dataset", "spam_hindi.csv")
    if os.path.exists(hi_path):
        df_hi = _read_csv_any(hi_path)
        datasets.append(_normalize_and_select(df_hi))

    if not datasets:
        raise FileNotFoundError("No datasets found. Expected at least dataset/spam.csv")

    combined = pd.concat(datasets, ignore_index=True)
    combined.dropna(subset=["text", "label_num"], inplace=True)
    return combined


def load_model() -> Pipeline:
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


# === Web scraping helpers ===
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SpamGuardBot/1.0)"}
translator = Translator()


# === Spam heuristics (ported from app) ===
def translate_to_english(text: str) -> str:
    try:
        return translator.translate(text, dest="en").text
    except Exception:
        return text


def is_obvious_spam(text_en: str) -> bool:
    text_l = text_en.lower()
    keywords = [
        "congratulations", "congrats", "won", "winner", "gift card", "giftcard",
        "amazon", "prize", "reward", "claim", "click here", "limited time",
        "urgent", "free",
        "$", "₹", "€", "৳",
        "बधाई", "विजेता", "जीता", "जीत", "उपहार", "गिफ्ट", "कार्ड", "इनाम",
        "पुरस्कार", "रिवॉर्ड", "दावा", "क्लिक करें", "यहां क्लिक करें", "अभी",
        "सीमित समय", "फ्री", "मुफ्त",
        "অভিনন্দন", "কনগ্রাচস", "জিতেছেন", "জয়ী", "বিজয়ী", "উপহার",
        "গিফট", "কার্ড", "পুরস্কার", "রিওয়ার্ড", "রিওয়ার্ড", "ক্লেইম",
        "এখানে ক্লিক", "এখনই", "সীমিত সময়", "ফ্রি"
    ]
    hits = sum(1 for k in keywords if k in text_l)
    won_combo_en = bool(re.search(r"\bwon\b.*\b(gift|prize|reward|card)\b", text_l))
    claim_here_en = bool(re.search(r"\bclaim\b.*\b(here|now)\b", text_l))
    won_combo_hi = bool(re.search(r"(जीत|जीता|विजेता).*(इनाम|उपहार|रिवॉर्ड|पुरस्कार|कार्ड)", text_l))
    claim_here_hi = bool(re.search(r"(दावा|क्लेम).*(अभी|अब|यहाँ|यहां)", text_l))
    won_combo_bn = bool(re.search(r"(জিতেছেন|বিজয়ী|জয়ী).*(পুরস্কার|উপহার|রিওয়ার্ড|কার্ড)", text_l))
    claim_here_bn = bool(re.search(r"(ক্লেইম|দাবি).*(এখনই|এখানে)", text_l))
    return hits >= 2 or won_combo_en or claim_here_en or won_combo_hi or claim_here_hi or won_combo_bn or claim_here_bn


def fetch_page(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text


def extract_text_blocks(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    texts = []
    # Common elements where examples live
    for selector in ["p", "li", "blockquote", "pre", "article", "div"]:
        for el in soup.select(selector):
            t = el.get_text(" ", strip=True)
            if not t:
                continue
            # Keep moderately sized snippets
            if 30 <= len(t) <= 600:
                texts.append(t)
    # Deduplicate
    uniq = []
    seen = set()
    for t in texts:
        k = re.sub(r"\s+", " ", t)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(k)
    return uniq


def scrape_samples(urls: List[str]) -> List[str]:
    samples = []
    for url in urls:
        try:
            html = fetch_page(url)
            blocks = extract_text_blocks(html)
            samples.extend(blocks)
            time.sleep(0.5)
        except Exception:
            continue
    return samples


def build_labeled_corpus(spam_urls: List[str], ham_urls: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    spam_samples = scrape_samples(spam_urls)
    ham_samples = scrape_samples(ham_urls)
    # Augment spam samples with Hindi and Bengali translations to ensure multilingual coverage
    spam_aug = []
    for t in spam_samples:
        spam_aug.append(t)
        try:
            hi = translator.translate(t, dest="hi").text
            bn = translator.translate(t, dest="bn").text
            spam_aug.extend([hi, bn])
        except Exception:
            pass
    df_spam = pd.DataFrame({"text": spam_aug, "label_num": 1})
    df_ham = pd.DataFrame({"text": ham_samples, "label_num": 0})
    return df_spam, df_ham


def evaluate_on_web(spam_urls: List[str], ham_urls: List[str]) -> None:
    model = load_model()
    df_spam, df_ham = build_labeled_corpus(spam_urls, ham_urls)
    df_eval = pd.concat([df_spam, df_ham], ignore_index=True)
    if df_eval.empty:
        print("No samples scraped. Please provide reachable URLs with examples.")
        return
    y_true = df_eval["label_num"].values
    texts = df_eval["text"].tolist()
    # Base predictions
    y_pred = model.predict(texts)
    # Rule-based lift to ensure obvious spam (including translations) gets marked as spam
    lifted = []
    for i, txt in enumerate(texts):
        if is_obvious_spam(translate_to_english(txt)):
            lifted.append(1)
        else:
            lifted.append(y_pred[i])
    y_pred = lifted
    acc = accuracy_score(y_true, y_pred)
    print(f"Samples: {len(df_eval)} | Accuracy: {acc:.3f}")
    print(classification_report(y_true, y_pred, target_names=["ham", "spam"]))
    df_eval["pred"] = y_pred
    df_eval.to_csv("web_eval_results.csv", index=False, encoding="utf-8-sig")
    print("Saved detailed results to web_eval_results.csv")


if __name__ == "__main__":
    # Provide your own URLs here. These should be pages that explicitly list
    # spam/phishing examples and legitimate email/message examples respectively.
    SPAM_URLS = [
        "https://blog.textingbase.com/how-to-identify-spam-text-messages",
        "https://www.pandasecurity.com/en/mediacenter/spam-text-message-examples/"
    ]
    HAM_URLS = [
        "https://www.aljazeera.com/",
        "https://www.bbc.com/",
        "https://www.nytimes.com/",
        "https://www.washingtonpost.com/",
        "https://www.theguardian.com/",
        "https://www.lemonde.fr/",
        "https://www.spiegel.de/",
        "https://www.faz.net/",
        "https://www.taz.de/",
        "https://www.sueddeutsche.de/",
    ]
    evaluate_on_web(SPAM_URLS, HAM_URLS)


