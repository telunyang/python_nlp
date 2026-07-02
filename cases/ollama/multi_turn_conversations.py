import os
import json
from ollama import Client


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ASSISTANT_ID = os.getenv("OLLAMA_ASSISTANT", "qwen3.5:0.8b")
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


def stream_chat(user_message, history):
    messages = build_ollama_messages(history)

    messages.append({
        "role": "user",
        "content": user_message
    })

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
            print(text, end="", flush=True)
            full_reply += text

    print()

    history.append({
        "role": "user",
        "text": user_message
    })

    history.append({
        "role": "assistant",
        "text": full_reply
    })

    save_history(history)


def main():
    chat_history = load_history()

    print(f"系統：已連線到 Ollama: {OLLAMA_HOST}")
    print(f"系統：目前 assistant: {ASSISTANT_ID}")
    print("系統：輸入 exit 或空白可結束；輸入 clear 可清除歷史對話。")

    while True:
        try:
            question = input("你：").strip()
        except KeyboardInterrupt:
            print("\n系統：收到中斷，對話結束。")
            break

        if question == "" or question == "exit":
            break

        if question == "clear":
            chat_history = []
            save_history(chat_history)
            print("系統：歷史對話已清除")
            continue

        print("機器人：", end="", flush=True)

        try:
            stream_chat(question, chat_history)
        except Exception as e:
            print()
            print(f"系統：呼叫 Ollama 時發生錯誤：{e}")
            print("系統：請確認 Ollama 已啟動，且 assistant 名稱正確。")

    print(f"對話結束，歷史已儲存到 {HISTORY_FILE}")


if __name__ == "__main__":
    main()
