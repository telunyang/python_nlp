# 測試 ollama 聊天功能

import asyncio
from ollama import AsyncClient
import time

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
        host='http://localhost:11434',
        timeout=600
    )
    response = await client.chat(
        model='qwen3.5:0.8b', 
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




# 測試 ollama 聊天功能 (streaming)
# async def chat():
#     t1 = time.time()

#     client = AsyncClient(
#         host='http://localhost:11434',
#         timeout=600
#     )
#     response = await client.chat(
#         model='qwen3.5:0.8b', 
#         messages=messages,
#         keep_alive="1h",
#         think=False,
#         stream=True
#     )
#     async for part in response:
#         print(part['message']['content'], end='', flush=True)

#     t2 = time.time()
#     print(f"Response time: {t2 - t1:.2f} seconds")

if __name__ == '__main__':
    asyncio.run(chat())