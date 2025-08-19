import os
import uuid
import json
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import google.generativeai as genai
from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------- Config ----------
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

genai.configure(api_key=API_KEY)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
SYSTEM_INSTRUCTION = os.getenv(
    "SYSTEM_INSTRUCTION",
    "You are a warm, consistent assistant. Maintain the user's preferred rhythm and mood across sessions. "
    "Be concise unless they ask for depth."
)

HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "200"))  # safety cap

# ---------- Model ----------
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    system_instruction=SYSTEM_INSTRUCTION
)

# ---------- Flask ----------
app = Flask(__name__, static_folder="static", static_url_path="")
app.secret_key = os.getenv("SECRET_KEY", "supersecret")  # set in Heroku
CORS(app, supports_credentials=True)

# Make session cookies last longer (e.g., 180 days)
app.config["PERMANENT_SESSION_LIFETIME"] = 60 * 60 * 24 * 180
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# In production behind HTTPS, consider:
# app.config["SESSION_COOKIE_SECURE"] = True

# ---------- Database ----------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///chatbot.db")
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class Conversation(Base):
    __tablename__ = "conversations"
    user_id = Column(String, primary_key=True)
    history = Column(Text)  # JSON: [{"role": "user"|"assistant", "content": "..."}]

Base.metadata.create_all(engine)

# ---------- Helpers ----------
def _normalize_history(history_list):
    """Convert our stored history to Gemini format."""
    norm = []
    for turn in history_list:
        role = (turn.get("role") or "").lower()
        content = turn.get("content") or ""
        if not content:
            continue
        gemini_role = "user" if role in ("user", "human") else "model"
        norm.append({"role": gemini_role, "parts": [content]})
    return norm

def _get_user_id():
    """Assign persistent cookie user_id."""
    session.permanent = True
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
        session.modified = True
    return session["user_id"]

def _load_history(user_id):
    db = SessionLocal()
    try:
        row = db.query(Conversation).filter_by(user_id=user_id).first()
        if not row or not row.history:
            return []
        return json.loads(row.history)
    except Exception:
        return []
    finally:
        db.close()

def _save_history(user_id, history):
    # Cap length to avoid unbounded growth
    if len(history) > HISTORY_MAX_TURNS:
        history = history[-HISTORY_MAX_TURNS:]

    db = SessionLocal()
    try:
        row = db.query(Conversation).filter_by(user_id=user_id).first()
        payload = json.dumps(history)
        if row:
            row.history = payload
        else:
            row = Conversation(user_id=user_id, history=payload)
            db.add(row)
        db.commit()
    finally:
        db.close()

# ---------- Routes ----------
@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})

@app.post("/api/chat")
def chat():
    """
    Body:
    {
      "message": "Hello",
      "temperature": 0.7,
      "max_output_tokens": 1024
    }
    """
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "message is required"}), 400

    user_id = _get_user_id()
    history = _load_history(user_id)
    gemini_history = _normalize_history(history)

    temperature = float(data.get("temperature", 0.7))
    max_output_tokens = int(data.get("max_output_tokens", 1024))

    try:
        chat_session = model.start_chat(history=gemini_history)
        resp = chat_session.send_message(
            user_message,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens
            }
        )

        reply = getattr(resp, "text", None)
        if not reply and getattr(resp, "candidates", None):
            # Fallback parse
            parts = resp.candidates[0].content.parts if resp.candidates[0].content else []
            reply = parts[0].text if parts else ""
        if not reply:
            reply = "(No response generated.)"

        # persist the long single-thread history
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})
        _save_history(user_id, history)

        return jsonify({"text": reply, "model": MODEL_NAME, "user_id": user_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/api/reset")
def reset():
    """Clear this user's persistent conversation."""
    user_id = _get_user_id()
    db = SessionLocal()
    try:
        row = db.query(Conversation).filter_by(user_id=user_id).first()
        if row:
            row.history = "[]"
            db.commit()
        return jsonify({"ok": True})
    finally:
        db.close()

# Optional: serve your UI at /static/index.html
@app.get("/")
def index():
    if os.path.exists(os.path.join(app.static_folder, "index.html")):
        return send_from_directory(app.static_folder, "index.html")
    return "Backend is running. Add your UI to /static/index.html", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
