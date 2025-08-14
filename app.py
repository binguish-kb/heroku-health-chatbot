import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai

# --- Config ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
SECRET_KEY = os.getenv("SECRET_KEY", "9f4c68f79e1a4d89aab72e92398cb3651f71eaf93a6b0f9fbc5f83dc0e3a6d2e
")
# Dataset config
DATA_PATH = os.getenv("DATA_PATH", "no_gpu_limit_500 1.json")
K = int(os.getenv("K", "3"))  # how many top matches to include as context

_vectorizer = None
_matrix = None
_raw_items = None
_display_blobs = None  # what we’ll show Gemini as "context blocks"

def _item_to_text_blob(item):
    """Make a searchable text blob from many possible JSON schemas."""
    if isinstance(item, dict):
        parts = []
        # try common keys; fallback to all values
        for key in ("question", "q", "title", "heading", "context", "content", "text", "body", "description"):
            if key in item and item[key]:
                parts.append(str(item[key]))
        if not parts:
            # join all values as fallback
            parts = [str(v) for v in item.values() if v is not None]
        blob = " ".join(parts)
        if not blob.strip():
            blob = json.dumps(item, ensure_ascii=False)
        return blob
    # non-dict rows
    return str(item)

def _item_to_context_block(item):
    if isinstance(item, dict):
        q = item.get("question") or item.get("q") or item.get("title") or ""
        a = item.get("answer") or item.get("a") or item.get("response") or item.get("content") or item.get("text") or ""
        ctx = item.get("context") or ""
        lines = []
        if ctx: lines.append(str(ctx))
        if q or a: lines.append(f"Q: {q}\nA: {a}")
        block = "\n".join([ln for ln in lines if ln.strip()])
        return block if block.strip() else json.dumps(item, ensure_ascii=False)
    return str(item)

def load_dataset():
    global _vectorizer, _matrix, _raw_items, _display_blobs
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            _raw_items = json.load(f)
        if not isinstance(_raw_items, list):
            # ensure list
            _raw_items = [_raw_items]
    except Exception as e:
        print(f"[WARN] Could not load dataset from {DATA_PATH}: {e}")
        _raw_items = []

    corpus = [_item_to_text_blob(it) for it in _raw_items]
    _display_blobs = [_item_to_context_block(it) for it in _raw_items]

    if corpus:
        _vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        _matrix = _vectorizer.fit_transform(corpus)
        print(f"[INFO] Indexed {len(corpus)} items from {DATA_PATH}.")
    else:
        _vectorizer, _matrix = None, None
        print("[INFO] No dataset items loaded; running pure Gemini mode.")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Set it with: heroku config:set GEMINI_API_KEY=...")

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

load_dataset()
def retrieve_top_k(query, k=K):
    if not query or _vectorizer is None or _matrix is None:
        return []
    qv = _vectorizer.transform([query])
    sims = cosine_similarity(qv, _matrix).ravel()
    if sims.size == 0:
        return []
    idxs = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i])) for i in idxs]

MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))


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

    # Retrieve top-K context blocks from dataset
    pairs = retrieve_top_k(user_message, K)
    context_blocks = []
    for idx, score in pairs:
        # You can filter by a minimum similarity if you want, e.g., score > 0.05
        context_blocks.append(_display_blobs[idx])
    context_text = "\n\n---\n\n".join([b for b in context_blocks if b.strip()])

    # Guardrail instruction to avoid hallucinations
    system_hint = (
        "You are a helpful assistant. Answer the user using ONLY the information in the CONTEXT. "
        "If the answer is not in the CONTEXT, say: 'I don't know based on the provided data.' "
        "Do not fabricate details."
    )

    # Build the prompt that includes CONTEXT + the latest user message
    grounded_prompt = f"{system_hint}\n\nCONTEXT:\n{context_text or '(no relevant context)'}\n\nUser: {user_message}\nAnswer:"
    try:
        # Keep using a chat session so the bot remembers the last few messages
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(grounded_prompt)
        bot_reply = getattr(response, "text", None) or "Sorry, I couldn't generate a response."
    except Exception as e:
        bot_reply = f"Error: {e}"

    # Update session history (last N messages)
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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
