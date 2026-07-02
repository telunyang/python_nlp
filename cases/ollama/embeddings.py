# 取得文本的嵌入向量

import asyncio
from ollama import AsyncClient
import time


async def embed_text():
    t1 = time.time()

    client = await AsyncClient(
        host='http://localhost:11434',
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