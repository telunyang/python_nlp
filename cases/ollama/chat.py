# 測試 ollama 聊天功能
import asyncio
from ollama import AsyncClient
import time


# 設定 Ollama API 的 URL
OLLAMA_HOST = "http://localhost:11434"
# OLLAMA_HOST = "https://bc92-136-66-88-68.ngrok-free.app"
# OLLAMA_HOST = "https://{NGROK_URL}" # 如果使用 ngrok，請取消註解並替換 {NGROK_URL} 為實際的 ngrok URL

messages = [
    {
        'role': 'user', 
        'content': '幸福的定義是什麼？簡單說明。'
    },
]

# 測試 ollama 聊天功能
async def chat():
    t1 = time.time()

    client = AsyncClient(
        host=OLLAMA_HOST,
        timeout=600
    )
    response = await client.chat(
        model='gemma4:e2b', # gpt-oss:20b # qwen3.5:0.8b # gemma4:e2b 
        messages=messages,
        keep_alive="1h",
        think=False,
        options={
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.95
        },
    )
    print(response.message.content)

    t2 = time.time()
    print(f"Response time: {t2 - t1:.2f} seconds")




# # 測試 ollama 聊天功能 (streaming)
# async def chat():
#     t1 = time.time()

#     client = AsyncClient(
#         host=OLLAMA_HOST,
#         timeout=600
#     )
#     response = await client.chat(
#         model='gemma4:e2b', # gpt-oss:20b # qwen3.5:0.8b # gemma4:e2b
#         messages=messages,
#         keep_alive="1h",
#         think=False,
#         stream=True,
#         options={
#             "temperature": 1.0,
#             "top_k": 64,
#             "top_p": 0.95
#         },
#     )
#     async for part in response:
#         print(part['message']['content'], end='', flush=True)

#     t2 = time.time()
#     print(f"Response time: {t2 - t1:.2f} seconds")

if __name__ == '__main__':
    asyncio.run(chat())