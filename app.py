import os
import json
import glob
import re
import string
from typing import List, Dict, Tuple

from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- translation / language detection ----
from langdetect import detect_langs
from deep_translator import GoogleTranslator

# =========================
# Configuration (env vars)
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Session + memory
SECRET_KEY  = os.getenv("SECRET_KEY")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))  # total (user+bot) messages kept

# Retrieval settings
K        = int(os.getenv("K", "3"))          # top-k context chunks
MIN_SIM  = float(os.getenv("MIN_SIM", "0.05"))  # cosine similarity threshold

# Dataset locations (glob for many CSVs; DATA_PATH for one CSV/JSON)
DATA_GLOB = os.getenv("DATA_GLOB", "*.csv")
DATA_PATH = os.getenv("DATA_PATH", "")

# Optional explicit CSV column hinting
CSV_Q_COL = os.getenv("CSV_Q_COL", "")  # e.g. "question"
CSV_A_COL = os.getenv("CSV_A_COL", "")  # e.g. "answer"

# Multilingual toggles
TRANSLATION_ENABLED = os.getenv("TRANSLATION_ENABLED", "true").lower() in {"1","true","yes","on"}
MIN_DETECT_CONF     = float(os.getenv("MIN_DETECT_CONF", "0.80"))
SHORT_MSG_LEN       = int(os.getenv("SHORT_MSG_LEN", "4"))    # <=N chars → treat as English
FORCE_ENGLISH_ONLY  = os.getenv("FORCE_ENGLISH_ONLY", "false").lower() in {"1","true","yes","on"}

# Validate critical keys
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Run: heroku config:set GEMINI_API_KEY=...")

if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is not set. Run: heroku config:set SECRET_KEY=...")

# Flask app
app = Flask(__name__)
app.secret_key = SECRET_KEY

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

# Globals for retrieval
_vectorizer: TfidfVectorizer = None
_matrix = None
_raw_items: List[Dict] = []
_display_blobs: List[str] = []

# =========================
# Helpers: language / translation
# =========================
_EN_STOPWORDS = {
    "the","be","to","of","and","a","in","that","have","i","it","for","not","on",
    "with","he","as","you","do","at","this","but","his","by","from","we","they",
    "say","her","she","or","an","will","my","one","all","would","there","their"
}

def looks_english(text: str) -> bool:
    """Heuristic: short len → English; high ASCII ratio; >=2 English stopwords."""
    if not text:
        return True
    if len(text.strip()) <= SHORT_MSG_LEN:
        return True
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True
    ascii_letters = [ch for ch in letters if ch in string.ascii_letters]
    ascii_ratio = len(ascii_letters) / max(1, len(letters))
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    hits = sum(1 for t in tokens if t in _EN_STOPWORDS)
    return ascii_ratio >= 0.85 or hits >= 2

def detect_lang_confident(text: str, default="en", min_conf=0.80) -> str:
    try:
        langs = detect_langs(text)  # e.g. [Lang('en', 0.9999)]
        if not langs:
            return default
        top = langs[0]
        lang = top.lang
        prob = getattr(top, "prob", 0.0)
        return lang if prob >= min_conf else default
    except Exception:
        return default

def translate(text: str, src_lang: str, dest_lang: str) -> str:
    if not TRANSLATION_ENABLED or not text or src_lang == dest_lang or FORCE_ENGLISH_ONLY:
        return text
    try:
        return GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
    except Exception:
        return text

# =========================
# Helpers: dataset shaping
# =========================
def _item_to_text_blob(item: Dict) -> str:
    """Build a searchable text blob from many possible schemas."""
    if isinstance(item, dict):
        preferred = [
            "question", "q", "prompt", "title", "heading", "query", "input", "output",
            "answer", "a", "response", "content", "text", "body", "description", "context"
        ]
        parts = []
        for k in preferred:
            if k in item and item[k] is not None:
                parts.append(str(item[k]))
        if not parts:
            parts = [str(v) for v in item.values() if v is not None]
        blob = " ".join(parts).strip()
        return blob if blob else json.dumps(item, ensure_ascii=False)
    return str(item)

def _item_to_context_block(item: Dict) -> str:
    """Human-readable block that we show Gemini as CONTEXT."""
    if not isinstance(item, dict):
        return str(item)
    q = item.get("question") or item.get("q") or item.get("prompt") or item.get("title") or ""
    a = item.get("answer") or item.get("a") or item.get("response") or item.get("content") or item.get("text") or ""
    ctx = item.get("context") or item.get("description") or ""
    lines = []
    if ctx: lines.append(str(ctx))
    if q or a: lines.append(f"Q: {q}\nA: {a}")
    block = "\n".join([ln for ln in lines if ln.strip()])
    return block if block.strip() else _item_to_text_blob(item)

def _rows_from_csv(path: str) -> List[Dict]:
    """Read one CSV and map rows into dicts with best-guess Q/A fields."""
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, encoding="ISO-8859-1")

    df.columns = [str(c).strip().lower() for c in df.columns]

    q_col = CSV_Q_COL.lower() if CSV_Q_COL else None
    a_col = CSV_A_COL.lower() if CSV_A_COL else None

    if not q_col:
        for cand in ["question","q","prompt","title","query","input"]:
            if cand in df.columns: q_col = cand; break
    if not a_col:
        for cand in ["answer","a","response","content","text","reply","output"]:
            if cand in df.columns: a_col = cand; break

    rows: List[Dict] = []
    for _, r in df.iterrows():
        record: Dict = {}
        for c in df.columns:
            val = r.get(c)
            if pd.isna(val): continue
            record[c] = str(val)
        if q_col and q_col in df.columns:
            record.setdefault("question", str(r[q_col]))
        if a_col and a_col in df.columns:
            record.setdefault("answer", str(r[a_col]))
        rows.append(record)
    return rows

def load_dataset():
    """Load dataset from DATA_PATH (single) or DATA_GLOB (many CSVs), then build TF-IDF index."""
    global _vectorizer, _matrix, _raw_items, _display_blobs

    items: List[Dict] = []

    if DATA_PATH:
        p = DATA_PATH
        if p.lower().endswith(".json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else [data]
                print(f"[INFO] Loaded JSON items from {p}: {len(items)}")
            except Exception as e:
                print(f"[WARN] Could not load JSON from {p}: {e}")
        elif p.lower().endswith(".csv"):
            try:
                batch = _rows_from_csv(p)
                items.extend(batch)
                print(f"[INFO] Loaded {len(batch)} CSV rows from {p}")
            except Exception as e:
                print(f"[WARN] Failed to load CSV {p}: {e}")
        else:
            print(f"[WARN] Unsupported DATA_PATH extension: {p}")
    else:
        paths = sorted(glob.glob(DATA_GLOB))
        if not paths:
            print(f"[INFO] No CSV files matched {DATA_GLOB}. Running pure Gemini mode.")
        total = 0
        for p in paths:
            try:
                batch = _rows_from_csv(p)
                items.extend(batch)
                total += len(batch)
                print(f"[INFO] Loaded {len(batch)} rows from {p}")
            except Exception as e:
                print(f"[WARN] Failed to load {p}: {e}")
        if total:
            print(f"[INFO] Total CSV rows loaded: {total}")

    _raw_items = items
    corpus = [_item_to_text_blob(it) for it in _raw_items]
    _display_blobs = [_item_to_context_block(it) for it in _raw_items]

    if corpus:
        _vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        _matrix = _vectorizer.fit_transform(corpus)
        print(f"[INFO] Indexed {len(corpus)} items.")
    else:
        _vectorizer, _matrix = None, None
        print("[INFO] No dataset items loaded; fallback to pure Gemini.")

# Build index on startup
load_dataset()

# =========================
# Retrieval + chat helpers
# =========================
def retrieve_top_k(query: str, k: int) -> List[Tuple[int, float]]:
    """Return [(row_index, similarity), ...] for the top-k matches."""
    if not query or _vectorizer is None or _matrix is None:
        return []
    qv = _vectorizer.transform([query])
    sims = cosine_similarity(qv, _matrix).ravel()
    if sims.size == 0:
        return []
    idxs = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i])) for i in idxs]

# =========================
# Routes
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please type a message."}), 400

    # Language handling (heuristics-first)
    if looks_english(user_message):
        user_lang = "en"
    else:
        user_lang = detect_lang_confident(user_message, default="en", min_conf=MIN_DETECT_CONF)

    # For retrieval, translate to English only if confidently non-English
    user_message_en = user_message if user_lang == "en" else translate(user_message, src_lang=user_lang, dest_lang="en")

    # Short-term history from session
    history = session.get("history", [])

    # Retrieve relevant context from dataset using the English version
    pairs = retrieve_top_k(user_message_en, K)  # [(idx, sim), ...]
    valid_pairs = [(i, s) for (i, s) in pairs if s >= MIN_SIM]
    use_dataset = len(valid_pairs) > 0

    if use_dataset:
        # Build CONTEXT from valid matches (English snippets)
        context_blocks = [_display_blobs[idx] for idx, _ in valid_pairs]
        context_text_en = "\n\n---\n\n".join([b for b in context_blocks if b.strip()])

        system_hint = (
            "You are a helpful health assistant. Answer the user using ONLY the information in the CONTEXT. "
            "If the answer is not in the CONTEXT, say: 'I don't know based on the provided data.' "
            "Do not fabricate details."
        )
        grounded_prompt_en = f"{system_hint}\n\nCONTEXT:\n{context_text_en}\n\nUser: {user_message_en}\nAnswer:"

        try:
            chat_session = model.start_chat(history=history)
            response = chat_session.send_message(grounded_prompt_en)
            bot_reply_en = getattr(response, "text", "") or ""
        except Exception as e:
            # Graceful handling (e.g., quota 429)
            msg = str(e)
            if "429" in msg or "quota" in msg.lower():
                bot_reply_en = "I'm temporarily over my request limit. Please try again later."
            else:
                bot_reply_en = f"Error: {e}"

        # If dataset-based reply still doesn't answer, fall back to open model
        normalized = bot_reply_en.lower().strip()
        if (
            "i don't know based on the provided data" in normalized
            or normalized in {"i don't know.", "i don't know", "idk"}
            or len(normalized) < 8
        ):
            try:
                response2 = chat_session.send_message(user_message_en)
                gen_reply_en = getattr(response2, "text", "") or "Sorry, I couldn't generate a response."
            except Exception as e:
                msg = str(e)
                if "429" in msg or "quota" in msg.lower():
                    gen_reply_en = "I'm temporarily over my request limit. Please try again later."
                else:
                    gen_reply_en = f"Error: {e}"
            # No prefix, no “Gemini” signature
            bot_reply = gen_reply_en if user_lang == "en" else translate(gen_reply_en, src_lang="en", dest_lang=user_lang)
        else:
            # Translate final dataset-grounded answer back to the user's language
            bot_reply = bot_reply_en if user_lang == "en" else translate(bot_reply_en, src_lang="en", dest_lang=user_lang)

    else:
        # No good matches → fallback to open model (English), then translate back
        try:
            chat_session = model.start_chat(history=history)
            response = chat_session.send_message(user_message_en)
            gen_reply_en = getattr(response, "text", None) or "Sorry, I couldn't generate a response."
        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower():
                gen_reply_en = "I'm temporarily over my request limit. Please try again later."
            else:
                gen_reply_en = f"Error: {e}"

        # No prefix, no “Gemini” signature
        bot_reply = gen_reply_en if user_lang == "en" else translate(gen_reply_en, src_lang="en", dest_lang=user_lang)

    # Update session history (store original message + localized reply)
    history.append({"role": "user", "parts": [user_message]})
    history.append({"role": "model", "parts": [bot_reply]})
    session["history"] = history[-MAX_HISTORY:]

    return jsonify({"response": bot_reply})

@app.route("/reset", methods=["POST"])
def reset():
    session["history"] = []
    return jsonify({"ok": True})

@app.route("/healthz")
def healthz():
    return "ok", 200

if __name__ == "__main__":
    # For local debugging; Heroku will use gunicorn via Procfile
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
