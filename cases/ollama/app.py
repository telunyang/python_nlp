import os
import json

from flask import Flask, request, Response, render_template
from ollama import Client


app = Flask(__name__)

# Ollama 設定
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ASSISTANT_ID = os.getenv("OLLAMA_ASSISTANT", "gemma4:e2b")
HISTORY_FILE = "chat_history.json"

client = Client(
    host=OLLAMA_HOST,
    timeout=600
)


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def build_ollama_messages(history):
    messages = []

    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["text"]
        })

    return messages


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
    messages = build_ollama_messages(history)

    messages.append({
        "role": "user",
        "content": user_message
    })

    def generate():
        full_reply = ""

        stream = client.chat(
            model=ASSISTANT_ID,
            messages=messages,
            keep_alive="1h",
            stream=True
        )

        for part in stream:
            text = part.get("message", {}).get("content", "")

            if text:
                full_reply += text
                yield text

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
