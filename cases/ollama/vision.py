import asyncio
from ollama import AsyncClient
import time

# image_path = "D:\\your\\absolute\\path\\example01.jpg"
image_path = './example01.jpg'

# 設定 Ollama API 的 URL
OLLAMA_HOST = "http://localhost:11434"
# OLLAMA_HOST = "https://bc92-136-66-88-68.ngrok-free.app"
# OLLAMA_HOST = "https://{NGROK_URL}" # 如果使用 ngrok，請取消註解並替換 {NGROK_URL} 為實際的 ngrok URL

messages = [
    {
        'role': 'user', 
        'content': '請描述這張圖片的內容。',
        'images': [image_path] # 可以多張圖片，格式為 List[str]，每個元素為圖片的路徑
    }
]

async def recognize():
    t1 = time.time()

    client = AsyncClient(
        host=OLLAMA_HOST,
        timeout=600
    )
    
    response = await client.chat(
        model='gemma4:e2b', 
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




# async def recognize_stream():
#     t1 = time.time()

#     client = AsyncClient(
#         host='http://localhost:11434',
#         timeout=600
#     )
#     response = await client.chat(
#         model='gemma4:e2b', 
#         messages=messages,
#         keep_alive="1h",
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


if __name__ == "__main__":
    asyncio.run(recognize())