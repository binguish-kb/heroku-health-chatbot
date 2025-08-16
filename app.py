import os
import json
import glob
import re
import string
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from uuid import uuid4

from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Language / translation
from langdetect import detect_langs
from deep_translator import GoogleTranslator

# SQLite + optional encryption
import sqlite3
import base64
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # pip install cryptography
except Exception:
    AESGCM = None

# =========================
# Configuration (env vars)
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

SECRET_KEY  = os.getenv("SECRET_KEY", "dev-secret-change-me")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))

# Retrieval
K        = int(os.getenv("K", "3"))
MIN_SIM  = float(os.getenv("MIN_SIM", "0.12"))

# Dataset
DATA_GLOB = os.getenv("DATA_GLOB", "*.csv")
DATA_PATH = os.getenv("DATA_PATH", "")
CSV_Q_COL = os.getenv("CSV_Q_COL", "")
CSV_A_COL = os.getenv("CSV_A_COL", "")

# Multilingual
TRANSLATION_ENABLED = os.getenv("TRANSLATION_ENABLED", "true").lower() in {"1","true","yes","on"}
MIN_DETECT_CONF     = float(os.getenv("MIN_DETECT_CONF", "0.80"))
SHORT_MSG_LEN       = int(os.getenv("SHORT_MSG_LEN", "4"))
FORCE_ENGLISH_ONLY  = os.getenv("FORCE_ENGLISH_ONLY", "false").lower() in {"1","true","yes","on"}

# Journal / DB
SQLITE_PATH = os.getenv("SQLITE_PATH", "journal.sqlite")
ENC_KEY_RAW = os.getenv("ENC_KEY", "")  # optional 32-byte base64 or 64-char hex

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")

# Flask
app = Flask(__name__)
app.secret_key = SECRET_KEY

# Gemini
genai.configure(api_key=GEMINI_API_KEY)
GEN_CONFIG = genai.GenerationConfig(
    temperature=0.2, top_p=0.9, top_k=40, max_output_tokens=1024
)
model = genai.GenerativeModel(GEMINI_MODEL, generation_config=GEN_CONFIG)

# Retrieval globals
_vectorizer: Optional[TfidfVectorizer] = None
_matrix = None
_raw_items: List[Dict] = []
_display_blobs: List[str] = []

# =========================
# Language helpers
# =========================
_EN_STOPWORDS = {
    "the","be","to","of","and","a","in","that","have","i","it","for","not","on",
    "with","he","as","you","do","at","this","but","his","by","from","we","they",
    "say","her","she","or","an","will","my","one","all","would","there","their"
}

def looks_english(text: str) -> bool:
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
        langs = detect_langs(text)
        if not langs:
            return default
        top = langs[0]
        return top.lang if getattr(top, "prob", 0.0) >= min_conf else default
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
# Prompting helpers
# =========================
SYSTEM_CORE = (
    "You are a careful, supportive health information assistant.\n"
    "Provide general, educational guidance the user can apply now. Be specific and actionable.\n"
    "Use a calm, reassuring tone. Do not diagnose diseases or claim to be a clinician.\n"
    "NEVER say 'I am an AI' or give generic boilerplate disclaimers.\n"
    "If important info is missing, begin with 2–4 targeted follow-up questions before guidance.\n"
    "If the user's name is known, address them once naturally at the start.\n"
    "Structure EVERY answer with these sections:\n"
    "1) Summary — 1–2 sentences restating the concern.\n"
    "2) What it might be — general possibilities (not a diagnosis).\n"
    "3) Clinical reasoning outline — factors that make options more/less likely; what else to ask/check; simple risk signals.\n"
    "4) What you can do now — concrete self-care (non-drug first; dosing ranges if appropriate; metric units; hygiene/isolation when relevant).\n"
    "5) When to see a professional — routine triggers (e.g., >48–72h, worsening, special populations).\n"
    "6) Red flags — urgent symptoms requiring immediate/urgent care.\n"
)

def build_system_hint(lang_code: str, name: str = "") -> str:
    name_line = f"If a name is available (e.g., '{name}'), greet them once." if name else "Greet the user once without a name if unknown."
    return (
        SYSTEM_CORE +
        f"\nAlways answer in language code '{lang_code}'. "
        "Be concise but complete. Avoid generic apologies.\n" +
        name_line
    )

GENERIC_MARKERS = [
    "i am an ai", "i'm an ai", "cannot give medical advice",
    "consult a doctor", "consult your doctor", "seek medical attention",
    "as an ai", "i cannot provide medical advice"
]

def looks_generic(text: str) -> bool:
    if not text:
        return True
    t = text.lower().strip()
    if len(t) < 20:
        return True
    return any(m in t for m in GENERIC_MARKERS)

def build_grounded_prompt(context_text_en: str, user_message_en: str, lang: str, name: str) -> str:
    system_hint = build_system_hint(lang, name)
    return f"{system_hint}\n\nCONTEXT:\n{context_text_en}\n\nUser: {user_message_en}\nAnswer:"

def build_open_prompt(user_message_en: str, lang: str, name: str) -> str:
    system_hint = build_system_hint(lang, name)
    return f"{system_hint}\n\nUser: {user_message_en}\nAnswer:"

# =========================
# Dataset shaping
# =========================
def _item_to_text_blob(item: Dict) -> str:
    if isinstance(item, dict):
        preferred = [
            "question","q","prompt","title","heading","query","input","output",
            "answer","a","response","content","text","body","description","context"
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
    global _vectorizer, _matrix, _raw_items, _display_blobs
    items: List[Dict] = []

    if DATA_PATH:
        p = DATA_PATH
        if p.lower().endswith(".json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else [data]
            except Exception as e:
                print(f"[WARN] Could not load JSON from {p}: {e}")
        elif p.lower().endswith(".csv"):
            try:
                items.extend(_rows_from_csv(p))
            except Exception as e:
                print(f"[WARN] Failed to load CSV {p}: {e}")
        else:
            print(f"[WARN] Unsupported DATA_PATH: {p}")
    else:
        paths = sorted(glob.glob(DATA_GLOB))
        if not paths:
            print(f"[INFO] No CSV files matched {DATA_GLOB}. Running pure Gemini mode.")
        for p in paths:
            try:
                items.extend(_rows_from_csv(p))
            except Exception as e:
                print(f"[WARN] Failed to load {p}: {e}")

    _raw_items = items
    corpus = [_item_to_text_blob(it) for it in _raw_items]
    _display_blobs = [_item_to_context_block(it) for it in _raw_items]

    if corpus:
        _vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        _matrix = _vectorizer.fit_transform(corpus)
        print(f"[INFO] Indexed {len(corpus)} items.")
    else:
        _vectorizer, _matrix = None, None
        print("[INFO] No dataset items loaded; fallback to pure Gemini.")

load_dataset()

# =========================
# SQLite + optional encryption
# =========================
def _load_aes_key():
    if not ENC_KEY_RAW or not AESGCM:
        return None
    try:
        if all(c in "0123456789abcdefABCDEF" for c in ENC_KEY_RAW) and len(ENC_KEY_RAW) == 64:
            return bytes.fromhex(ENC_KEY_RAW)
        return base64.b64decode(ENC_KEY_RAW)
    except Exception:
        return None
AES_KEY = _load_aes_key()

def db():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS users (
          id TEXT PRIMARY KEY,
          name TEXT,
          dob TEXT, -- YYYY-MM-DD
          emergency_contact TEXT,
          created_at TEXT
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS memories (
          user_id TEXT,
          key TEXT,
          value TEXT,
          updated_at TEXT,
          PRIMARY KEY (user_id, key)
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS journal_entries (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT,
          ts TEXT,
          lang TEXT,
          text_original TEXT,
          text_en TEXT,
          tags TEXT,
          mood INTEGER,
          enc_blob BLOB
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT,
          ts TEXT,
          role TEXT, -- 'user' or 'assistant'
          text_en TEXT
        )""")
        conn.commit()
init_db()

def aes_encrypt(plaintext: str) -> Optional[bytes]:
    if not AES_KEY or not AESGCM:
        return None
    aad = b"journal-v1"
    aes = AESGCM(AES_KEY)
    nonce = os.urandom(12)
    ct = aes.encrypt(nonce, plaintext.encode("utf-8"), aad)
    return nonce + ct

# =========================
# Users / memory / journal
# =========================
def generate_user_id() -> str:
    return uuid4().hex[:8].upper()

def get_user_by_id(user_id: str):
    with db() as conn:
        r = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(r) if r else None

def create_user(name: str, dob: str, emergency_contact: str = None) -> str:
    uid = generate_user_id()
    with db() as conn:
        conn.execute(
            "INSERT INTO users (id,name,dob,emergency_contact,created_at) VALUES (?,?,?,?,?)",
            (uid, name.strip(), dob.strip(), (emergency_contact or ""), datetime.utcnow().isoformat()),
        )
        conn.commit()
    return uid

def verify_user(user_id: str, dob: str) -> bool:
    u = get_user_by_id(user_id)
    return bool(u and (u.get("dob") or "") == dob.strip())

def set_memory(user_id: str, key: str, value: str):
    with db() as conn:
        conn.execute("""
            INSERT INTO memories(user_id,key,value,updated_at)
            VALUES (?,?,?,?)
            ON CONFLICT(user_id,key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """, (user_id, key, value, datetime.utcnow().isoformat()))
        conn.commit()

def get_memory(user_id: str, key: str) -> str:
    with db() as conn:
        r = conn.execute("SELECT value FROM memories WHERE user_id=? AND key=?", (user_id, key)).fetchone()
        return r["value"] if r else ""

def purge_old_journal(user_id: str):
    cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
    with db() as conn:
        conn.execute("DELETE FROM journal_entries WHERE user_id=? AND ts < ?", (user_id, cutoff))
        conn.commit()

def add_journal_entry(user_id: str, lang: str, text_original: str, text_en: str, tags: str = "", mood: Optional[int] = None):
    enc_blob = aes_encrypt(text_en or text_original or "")
    with db() as conn:
        conn.execute("""
            INSERT INTO journal_entries(user_id, ts, lang, text_original, text_en, tags, mood, enc_blob)
            VALUES (?,?,?,?,?,?,?,?)
        """, (user_id, datetime.utcnow().isoformat(), lang, text_original, text_en, tags or "", mood, enc_blob))
        conn.commit()

def log_conversation(user_id: str, role: str, text_en: str):
    with db() as conn:
        conn.execute("""
            INSERT INTO conversations(user_id, ts, role, text_en)
            VALUES (?,?,?,?)
        """, (user_id, datetime.utcnow().isoformat(), role, text_en or ""))
        conn.commit()

def last_n_conversation(user_id: str, n: int = 8) -> str:
    with db() as conn:
        rows = conn.execute("""
            SELECT role, text_en FROM conversations
            WHERE user_id=? ORDER BY id DESC LIMIT ?
        """, (user_id, n)).fetchall()
    lines = []
    for r in reversed(rows):
        lines.append(f"{r['role']}: {r['text_en']}")
    return "\n".join(lines)

# =========================
# Crisis detection
# =========================
CRISIS_TERMS = [
    "suicide", "kill myself", "self-harm", "self harm", "overdose",
    "end my life", "hurting myself", "can't go on", "cant go on"
]
def detect_crisis(text_en: str) -> bool:
    t = (text_en or "").lower()
    return any(term in t for term in CRISIS_TERMS)

def supportive_resources(lang: str) -> str:
    return ("If you feel in immediate danger, call your local emergency number now.\n"
            "Consider reaching out to a trusted person, or a crisis helpline in your country.")

# =========================
# Retrieval
# =========================
def retrieve_top_k(query: str, k: int) -> List[Tuple[int, float]]:
    if not query or _vectorizer is None or _matrix is None:
        return []
    qv = _vectorizer.transform([query])
    sims = cosine_similarity(qv, _matrix).ravel()
    if sims.size == 0:
        return []
    idxs = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i])) for i in idxs]

# =========================
# Natural language intents
# =========================
EXPORT_PATTERNS = [
    r"\bexport (my )?journal\b",
    r"\bdownload (my )?journal\b",
    r"\bsend (me )?my journal\b"
]
SUMMARY_PATTERNS = [
    r"\bsummary of my week\b",
    r"\bsummarize my week\b",
    r"\bsummarise my week\b",
    r"\bweekly reflection\b",
    r"\breflect on this week\b"
]
EMERGENCY_SET_PAT = re.compile(r"(?:my\s+)?emergency\s+contact\s+(?:is|=)\s*(.+)", re.IGNORECASE)
EMERGENCY_CLEAR_PAT = re.compile(r"(?:remove|clear|forget)\s+(?:my\s+)?emergency\s+contact", re.IGNORECASE)

def wants_export(msg: str) -> bool:
    s = (msg or "").lower()
    return any(re.search(p, s) for p in EXPORT_PATTERNS)

def wants_week_summary(msg: str) -> bool:
    s = (msg or "").lower()
    return any(re.search(p, s) for p in SUMMARY_PATTERNS)

# =========================
# Onboarding
# =========================
def start_onboarding():
    session["onboarding_stage"] = "ask_existing"
    return "Are you an existing user? (yes/no)"

def handle_onboarding(user_text: str) -> str:
    stage = session.get("onboarding_stage")
    t = user_text.strip().lower()

    if stage == "ask_existing":
        if t in ("yes","y"):
            session["onboarding_stage"] = "ask_id"
            return "Please enter your User ID."
        elif t in ("no","n"):
            session["onboarding_stage"] = "new_name"
            return "Welcome! Please tell me your full name."
        else:
            return "Please reply with 'yes' or 'no'."

    if stage == "ask_id":
        session["temp_user_id"] = user_text.strip().upper()
        session["onboarding_stage"] = "ask_dob"
        return "Please confirm your Date of Birth (YYYY-MM-DD)."

    if stage == "ask_dob":
        uid = session.get("temp_user_id","")
        dob = user_text.strip()
        if verify_user(uid, dob):
            session["user_id"] = uid
            session.pop("temp_user_id", None)
            session.pop("onboarding_stage", None)
            u = get_user_by_id(uid) or {}
            name = u.get("name") or ""
            return f"Welcome back{', ' + name if name else ''}! Your journal will be saved automatically."
        else:
            session.pop("temp_user_id", None)
            session["onboarding_stage"] = "ask_existing"
            return "I couldn’t verify those details. Are you an existing user? (yes/no)"

    if stage == "new_name":
        session["new_name"] = user_text.strip()
        session["onboarding_stage"] = "new_dob"
        return "Thanks. Please enter your Date of Birth (YYYY-MM-DD)."

    if stage == "new_dob":
        name = session.get("new_name","")
        dob = user_text.strip()
        uid = create_user(name=name, dob=dob)
        session["user_id"] = uid
        session.pop("new_name", None)
        session.pop("onboarding_stage", None)
        return f"Your account is set! Your User ID is {uid}. I’ll save your journal automatically."

    return start_onboarding()

# =========================
# Routes
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/journal/export", methods=["POST"])
def export_journal():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "Not verified. Please onboard first."}), 400
    with db() as conn:
        rows = conn.execute("""
            SELECT id, ts, lang, text_original, text_en, tags, mood
            FROM journal_entries WHERE user_id=? ORDER BY id DESC
        """, (uid,)).fetchall()
    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id","timestamp","lang","original","english","tags","mood"])
    for r in rows:
        writer.writerow([r["id"], r["ts"], r["lang"], r["text_original"], r["text_en"], r["tags"], r["mood"]])
    return jsonify({"filename":"journal.csv","content": buf.getvalue()})

@app.route("/reset", methods=["POST"])
def reset():
    session["history"] = []
    session.pop("user_id", None)
    session.pop("onboarding_stage", None)
    session.pop("temp_user_id", None)
    session.pop("new_name", None)
    return jsonify({"ok": True})

@app.route("/healthz")
def healthz():
    return "ok", 200

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please type a message."}), 400

    # Onboarding if needed
    if "user_id" not in session:
        if session.get("onboarding_stage"):
            reply = handle_onboarding(user_message)
            return jsonify({"response": reply})
        return jsonify({"response": start_onboarding()})

    user_id = session["user_id"]
    urec = get_user_by_id(user_id) or {}
    user_name = urec.get("name") or ""

    # Weekly cleanup
    purge_old_journal(user_id)

    # Language
    if looks_english(user_message):
        user_lang = "en"
    else:
        user_lang = detect_lang_confident(user_message, default="en", min_conf=MIN_DETECT_CONF)
    user_message_en = user_message if user_lang == "en" else translate(user_message, src_lang=user_lang, dest_lang="en")

    # Crisis detection
    crisis_prefix = ""
    if detect_crisis(user_message_en):
        crisis_prefix = supportive_resources(user_lang) + "\n\n"
        if not (urec.get("emergency_contact") or "").strip():
            crisis_prefix += "If you’d like, share an emergency contact (e.g., “My emergency contact is Sam, +61…”) and I’ll save it.\n\n"

    # Friendly natural emergency contact set/clear
    m_set = EMERGENCY_SET_PAT.search(user_message)
    if m_set:
        ec = m_set.group(1).strip()
        with db() as conn:
            conn.execute("UPDATE users SET emergency_contact=? WHERE id=?", (ec, user_id))
            conn.commit()
        ack = "Emergency contact saved."
        ack = ack if user_lang == "en" else translate(ack, "en", user_lang)
        # Journal the message then continue with health answer as usual
    elif EMERGENCY_CLEAR_PAT.search(user_message):
        with db() as conn:
            conn.execute("UPDATE users SET emergency_contact='' WHERE id=?", (user_id,))
            conn.commit()
        ack = "Emergency contact removed."
        ack = ack if user_lang == "en" else translate(ack, "en", user_lang)
    else:
        ack = ""

    # Always JOURNAL user input (original + English)
    add_journal_entry(
        user_id=user_id,
        lang=user_lang,
        text_original=user_message,
        text_en=user_message_en
    )

    # Natural “export my journal”
    if wants_export(user_message):
        with db() as conn:
            rows = conn.execute("""
                SELECT id, ts, lang, text_original, text_en, tags, mood
                FROM journal_entries WHERE user_id=? ORDER BY id DESC
            """, (user_id,)).fetchall()
        import csv, io
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id","timestamp","lang","original","english","tags","mood"])
        for r in rows:
            writer.writerow([r["id"], r["ts"], r["lang"], r["text_original"], r["text_en"], r["tags"], r["mood"]])
        csv_text = buf.getvalue()
        resp_msg = ("I’ve prepared your journal CSV below. "
                    "You can also POST to /journal/export to fetch it again.\n\n") + csv_text
        if crisis_prefix:
            resp_msg = crisis_prefix + resp_msg
        if ack:
            resp_msg = ack + "\n\n" + resp_msg
        # Log and return
        log_conversation(user_id, "user", user_message_en)
        log_conversation(user_id, "assistant", resp_msg if user_lang=="en" else translate(resp_msg,"en",user_lang))
        return jsonify({"response": resp_msg})

    # Natural “summarize my week”
    if wants_week_summary(user_message):
        with db() as conn:
            since = (datetime.utcnow() - timedelta(days=7)).isoformat()
            rows = conn.execute("""
                SELECT text_en FROM journal_entries
                WHERE user_id=? AND ts >= ?
                ORDER BY id ASC
            """, (user_id, since)).fetchall()
        corpus = "\n\n".join([r["text_en"] for r in rows])
        if corpus.strip():
            sys = ("You are a reflective journaling assistant. Create a compassionate, concise weekly reflection: "
                   "key themes, emotions, wins, challenges, and 2–3 gentle next steps. Answer in the user's language.")
            chat_session = model.start_chat(history=[])
            resp = chat_session.send_message(f"{sys}\n\nEntries (English):\n{corpus}\n\nWrite the reflection now in {user_lang}.")
            weekly = getattr(resp, "text", "") or "I couldn’t produce a summary this time."
            out = ("Weekly reflection:\n" + weekly)
            if crisis_prefix:
                out = crisis_prefix + out
            if ack:
                out = ack + "\n\n" + out
            # Log and return
            log_conversation(user_id, "user", user_message_en)
            log_conversation(user_id, "assistant", out if user_lang=="en" else translate(out,"en",user_lang))
            return jsonify({"response": out})

    # ===== Health assistant (default) =====
    prior = last_n_conversation(user_id, n=6)
    if prior:
        user_message_en = f"(Previous context)\n{prior}\n\n(Current)\n{user_message_en}"

    pairs = retrieve_top_k(user_message_en, K)
    valid_pairs = [(i, s) for (i, s) in pairs if s >= MIN_SIM]
    use_dataset = len(valid_pairs) > 0
    history = session.get("history", [])

    if use_dataset:
        context_blocks = [_display_blobs[idx] for idx, _ in valid_pairs]
        context_text_en = "\n\n---\n\n".join([b for b in context_blocks if b.strip()])
        grounded_prompt_en = build_grounded_prompt(context_text_en, user_message_en, user_lang, user_name)
        try:
            chat_session = model.start_chat(history=history)
            response = chat_session.send_message(grounded_prompt_en)
            bot_reply_candidate = getattr(response, "text", "") or ""
        except Exception as e:
            msg = str(e)
            bot_reply_candidate = "Temporary rate limit. Please try again in a moment." if ("429" in msg or "quota" in msg.lower()) else f"Error: {e}"
        if looks_generic(bot_reply_candidate):
            try:
                chat_session = model.start_chat(history=history)
                response2 = chat_session.send_message(
                    grounded_prompt_en + "\n\nIMPORTANT: Avoid generic disclaimers. Provide specific steps and red flags tailored to the scenario."
                )
                bot_reply_candidate2 = getattr(response2, "text", "") or ""
                if not looks_generic(bot_reply_candidate2):
                    bot_reply_candidate = bot_reply_candidate2
            except Exception:
                pass
        bot_reply = bot_reply_candidate
        bot_reply_lang = user_lang
    else:
        open_prompt_en = build_open_prompt(user_message_en, user_lang, user_name)
        try:
            chat_session = model.start_chat(history=history)
            response = chat_session.send_message(open_prompt_en)
            bot_reply_candidate = getattr(response, "text", None) or ""
        except Exception as e:
            msg = str(e)
            bot_reply_candidate = "Temporary rate limit. Please try again in a moment." if ("429" in msg or "quota" in msg.lower()) else f"Error: {e}"
        if looks_generic(bot_reply_candidate):
            try:
                chat_session = model.start_chat(history=history)
                response2 = chat_session.send_message(
                    open_prompt_en + "\n\nIMPORTANT: Avoid generic disclaimers. Provide specific, step-by-step self-care advice and clear thresholds for seeking care."
                )
                bot_reply_candidate2 = getattr(response2, "text", "") or ""
                if not looks_generic(bot_reply_candidate2):
                    bot_reply_candidate = bot_reply_candidate2
            except Exception:
                pass
        bot_reply = bot_reply_candidate
        bot_reply_lang = user_lang

    # Update short session history for Gemini continuity
    history.append({"role": "user", "parts": [user_message]})
    history.append({"role": "model", "parts": [bot_reply]})
    session["history"] = history[-MAX_HISTORY:]

    # Final localization
    final_reply = bot_reply if bot_reply_lang == user_lang else translate(bot_reply, src_lang=bot_reply_lang, dest_lang=user_lang)
    if crisis_prefix:
        final_reply = crisis_prefix + final_reply
    if ack:
        final_reply = ack + "\n\n" + final_reply

    # Persist conversation (English) for personalization
    log_conversation(user_id, "user", user_message if user_lang=="en" else translate(user_message, user_lang, "en"))
    log_conversation(user_id, "assistant", final_reply if user_lang=="en" else translate(final_reply, user_lang, "en"))

    return jsonify({"response": final_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
