import os
from google import genai
from dotenv import load_dotenv
load_dotenv()

# 讀取 API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 建立 Client
client = genai.Client(api_key=GOOGLE_API_KEY)

# 建立對話
chat = client.chats.create(
    model="gemini-1.5-flash-8b"
)

# 多輪對話 - 一次性
while True:
    # 輸入問題
    question = input("你：")

    # 結束對話
    if question == "exit":
        break

    # 輸入問題
    response = chat.send_message(question)
    print(f"機器人：{response.candidates[0].content.parts[0].text}", end="")


'''
# 多輪對話 - 串流式
while True:
    # 輸入問題
    question = input("你：")

    # 結束對話
    if question == "exit":
        break

    # 輸入問題
    response = chat.send_message_stream(question)
    print("機器人：", end="")
    for chunk in response:
        print(chunk.text, end="")
'''

# 輸出對話
for message in chat._curated_history:
    print(f'角色 - ', message.role, end=": ")
    print(message.parts[0].text)