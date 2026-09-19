import os
import json
from google import genai
from google.genai.types import UserContent, ModelContent, Part
from dotenv import load_dotenv
load_dotenv(override=True)

# 讀取 API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("找不到 GOOGLE_API_KEY，請確認 .env 是否有設定")

# 建立 Client
client = genai.Client(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemma-4-26b-a4b-it"
HISTORY_FILE = "chat_history.json"


# 讀取歷史對話
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# 儲存歷史對話
def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# 將 JSON 格式轉成 Gemini chat 需要的格式
def build_gemini_history(history):
    gemini_history = []

    for message in history:
        role = message["role"]
        text = message["text"]

        if role == "user":
            gemini_history.append(
                UserContent(parts=[Part(text=text)])
            )
        elif role == "model":
            gemini_history.append(
                ModelContent(parts=[Part(text=text)])
            )

    return gemini_history


# 讀取之前儲存的對話
chat_history = load_history()

# 建立對話
chat = client.chats.create(
    model=MODEL_NAME,
    history=build_gemini_history(chat_history)
)


while True:
    question = input("你：").strip()

    if question == "" or question == "exit":
        break

    if question == "clear":
        chat_history = []
        save_history(chat_history)

        chat = client.chats.create(
            model=MODEL_NAME,
            history=[]
        )

        print("系統：歷史對話已清除")
        continue

    print("機器人：", end="", flush=True)

    full_reply = ""

    # 串流式輸出
    response = chat.send_message_stream(question)

    for chunk in response:
        if chunk.text:
            print(chunk.text, end="", flush=True)
            full_reply += chunk.text

    print()

    # 將本輪對話加入歷史
    chat_history.append({
        "role": "user",
        "text": question
    })

    chat_history.append({
        "role": "model",
        "text": full_reply
    })

    # 每一輪都立刻儲存
    save_history(chat_history)


print("對話結束，歷史已儲存到 chat_history.json")