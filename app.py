import os
import json
import glob
from typing import List, Dict, Tuple

from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Configuration (env vars)
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Session + memory
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))  # total messages kept (user+bot), 10 ~= last 5 exchanges

# Retrieval settings
K = int(os.getenv("K", "3"))  # how many top items to include as context
MIN_SIM = float(os.getenv("MIN_SIM", "0.05"))  # similarity threshold (0.00–1.00)
FALLBACK_PREFIX = os.getenv(
    "FALLBACK_PREFIX",
    "This question isn’t covered by the provided data, but here’s an answer from Gemini:"
)

# Dataset locations
# Use DATA_GLOB to load multiple CSVs (e.g., data/*.csv), or DATA_PATH for a single file (csv or json).
DATA_GLOB = os.getenv("DATA_GLOB", "*.csv")
DATA_PATH = os.getenv("DATA_PATH", "")  # if set, takes priority over DATA_GLOB

# Optional explicit CSV column hinting
CSV_Q_COL = os.getenv("CSV_Q_COL", "")  # e.g., "question"
CSV_A_COL = os.getenv("CSV_A_COL", "")  # e.g., "answer"

# Validate key
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Run: heroku config:set GEMINI_API_KEY=...")

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
# Helpers: dataset shaping
# =========================
def _item_to_text_blob(item: Dict) -> str:
    """Build a searchable text blob from many possible schemas."""
    if isinstance(item, dict):
        preferred_keys = [
            "question", "q", "prompt", "title", "heading", "query", "input", "output"
            "answer", "a", "response", "content", "text", "body", "description", "context"
        ]
        parts = []
        for k in preferred_keys:
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
    if ctx:
        lines.append(str(ctx))
    if q or a:
        lines.append(f"Q: {q}\nA: {a}")
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
        for cand in ["question", "q", "prompt", "title", "query", "input"]:
            if cand in df.columns:
                q_col = cand
                break
    if not a_col:
        for cand in ["answer", "a", "response", "content", "text", "reply", "output"]:
            if cand in df.columns:
                a_col = cand
                break

    rows: List[Dict] = []
    for _, r in df.iterrows():
        record: Dict = {}
        for c in df.columns:
            val = r.get(c)
            if pd.isna(val):
                continue
            record[c] = str(val)
        if q_col and q_col in df.columns:
            record.setdefault("question", str(r[q_col]))
        if a_col and a_col in df.columns:
            record.setdefault("answer", str(r[a_col]))
        rows.append(record)
    return rows


def load_dataset():
    """Load dataset from DATA_PATH (single file) or DATA_GLOB (many CSVs), then build TF-IDF index."""
    global _vectorizer, _matrix, _raw_items, _display_blobs

    items: List[Dict] = []

    if DATA_PATH:
        # Single file mode
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
        # Multi-file CSV mode
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

    # Short-term history from session
    history = session.get("history", [])

    # Retrieve relevant context from dataset
    pairs = retrieve_top_k(user_message, K)  # [(idx, sim), ...]
    valid_pairs = [(i, s) for (i, s) in pairs if s >= MIN_SIM]
    use_dataset = len(valid_pairs) > 0

    if use_dataset:
        # Build CONTEXT from valid matches
        context_blocks = [_display_blobs[idx] for idx, _ in valid_pairs]
        context_text = "\n\n---\n\n".join([b for b in context_blocks if b.strip()])

        system_hint = (
            "You are a helpful health assistant. Answer the user using ONLY the information in the CONTEXT. "
            
            "Do not fabricate details."
        )
        grounded_prompt = f"{system_hint}\n\nCONTEXT:\n{context_text}\n\nUser: {user_message}\nAnswer:"

        try:
            chat_session = model.start_chat(history=history)
            response = chat_session.send_message(grounded_prompt)
            bot_reply = getattr(response, "text", None) or "Sorry, I couldn't generate a response."
        except Exception as e:
            bot_reply = f"Error: {e}"
    else:
        # Fallback to open Gemini answer (still keep memory)
        try:
            chat_session = model.start_chat(history=history)
            response = chat_session.send_message(user_message)
            gen_reply = getattr(response, "text", None) or "Sorry, I couldn't generate a response."
        except Exception as e:
            gen_reply = f"Error: {e}"
        bot_reply = f"{gen_reply}"

    # Update session history (trim to last N messages)
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
