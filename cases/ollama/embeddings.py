# 取得文本的嵌入向量

import asyncio
from ollama import AsyncClient
import time

# 設定 Ollama API 的 URL
OLLAMA_HOST = "http://localhost:11434"
# OLLAMA_HOST = "https://112c-136-66-88-68.ngrok-free.app"
# OLLAMA_HOST = "https://{NGROK_URL}" # 如果使用 ngrok，請取消註解並替換 {NGROK_URL} 為實際的 ngrok URL

async def embed_text():
    t1 = time.time()

    client = await AsyncClient(
        host=OLLAMA_HOST,
        timeout=600,
    )
    response = await client.embed(
        model='bge-m3',
        input='早安', # 你也可以放一句以上，例如 input=['早安', '早上好']，就會得到兩句話的嵌入向量
        keep_alive='1h',
    )

    print(response.embeddings[0])

    t2 = time.time()

    print(f'執行時間: {t2 - t1:.2f} 秒')


if __name__ == '__main__':
    asyncio.run(embed_text())