import os
import json
from flask import Flask, request, Response, render_template
from google import genai
from google.genai.types import UserContent, ModelContent, Part
from dotenv import load_dotenv

load_dotenv(override=True)


app = Flask(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASSISTANT_ID = os.getenv("GEMINI_ASSISTANT", "gemma-4-26b-a4b-it")
HISTORY_FILE = "chat_history.json"

if not GOOGLE_API_KEY:
    raise ValueError("找不到 GOOGLE_API_KEY，請確認 .env 是否有設定")

client = genai.Client(api_key=GOOGLE_API_KEY)


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def build_gemini_history(history):
    gemini_history = []

    for msg in history:
        role = msg["role"]
        text = msg["text"]

        if role == "user":
            gemini_history.append(
                UserContent(parts=[Part(text=text)])
            )

        elif role == "assistant":
            # Gemini SDK 這裡需要 ModelContent，
            # 但 chat_history.json 內部仍然統一儲存 role: assistant。
            gemini_history.append(
                ModelContent(parts=[Part(text=text)])
            )

    return gemini_history


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/history")
def get_history():
    return load_history()


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json()
    user_message = data["message"].strip()

    if not user_message:
        return Response("", mimetype="text/plain")

    history = load_history()

    chat = client.chats.create(
        model=ASSISTANT_ID,
        history=build_gemini_history(history)
    )

    def generate():
        full_reply = ""

        response = chat.send_message_stream(user_message)

        for chunk in response:
            if chunk.text:
                full_reply += chunk.text
                yield chunk.text

        history.append({
            "role": "user",
            "text": user_message
        })

        history.append({
            "role": "assistant",
            "text": full_reply
        })

        save_history(history)

    return Response(generate(), mimetype="text/plain")


@app.route("/api/clear", methods=["POST"])
def clear_history():
    save_history([])
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
