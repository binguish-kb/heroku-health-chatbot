# Gemini Chatbot (Heroku-ready)

A lightweight Flask web app with a modern chat UI (light-blue theme) that uses **Google Gemini** for responses and remembers the **last few messages** (short-term memory).

## Features
- Modern chat-style UI with bubbles, responsive layout
- Public but unlisted (no password screen)
- Short-term memory (default: last 5 exchanges)
- Ready for **Heroku** deployment

## Files
- `app.py` — Flask backend, Gemini integration, short-term memory via session cookies
- `templates/index.html` — Frontend UI, no external CSS/JS needed
- `requirements.txt` — Dependencies
- `Procfile` — Heroku web dyno entrypoint
- `runtime.txt` — Python version

## Configure & Deploy (Heroku)
```bash
# 1) Create app
heroku create my-gemini-chat

# 2) Set config (do NOT paste your key into code)
heroku config:set GEMINI_API_KEY="YOUR_API_KEY"
heroku config:set SECRET_KEY="a-long-random-string"   # used to sign session cookies
# Optional:
# heroku config:set GEMINI_MODEL="gemini-1.5-pro"     # default is gemini-1.5-flash
# heroku config:set MAX_HISTORY="10"                  # total messages kept (user+bot)

# 3) Push code (assuming your git remote is set to heroku)
git push heroku main
# or push to GitHub and connect Heroku to your repo

# 4) Visit the link provided by Heroku
heroku open
```

## Local Run (optional)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY="YOUR_API_KEY"
export SECRET_KEY="a-long-random-string"
python app.py
```

## Reset Conversation
POST `/reset` — clears stored short-term memory in the session.

---

**Note:** Keep the app **unlisted** by not advertising the URL. Anyone with the link can access it.
