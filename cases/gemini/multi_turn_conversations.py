import os
import json
from google import genai
from google.genai.types import UserContent, ModelContent, Part
from dotenv import load_dotenv

load_dotenv(override=True)


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
            gemini_history.append(
                ModelContent(parts=[Part(text=text)])
            )

    return gemini_history


def create_chat(history):
    return client.chats.create(
        model=ASSISTANT_ID,
        history=build_gemini_history(history)
    )


def main():
    chat_history = load_history()
    chat = create_chat(chat_history)

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
            chat = create_chat(chat_history)
            print("系統：歷史對話已清除")
            continue

        print("機器人：", end="", flush=True)

        full_reply = ""

        try:
            response = chat.send_message_stream(question)

            for chunk in response:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_reply += chunk.text

            print()

            chat_history.append({
                "role": "user",
                "text": question
            })

            chat_history.append({
                "role": "assistant",
                "text": full_reply
            })

            save_history(chat_history)

        except Exception as e:
            print()
            print(f"系統：呼叫 Gemini 時發生錯誤：{e}")

    print(f"對話結束，歷史已儲存到 {HISTORY_FILE}")


if __name__ == "__main__":
    main()
