import os
from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai

# --- Config ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Set it with: heroku config:set GEMINI_API_KEY=...")

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

# How many messages total to keep (user+bot). 10 = last 5 exchanges.
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please type a message."}), 400

    # Retrieve conversation history from the user's session cookie
    history = session.get("history", [])  # list of {"role": "user"/"model", "parts": [text]}

    # Start a chat session using recent history
    try:
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(user_message)
        bot_reply = getattr(response, "text", None) or "Sorry, I couldn't generate a response."
    except Exception as e:
        bot_reply = f"Error: {e}"

    # Update history and trim
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
