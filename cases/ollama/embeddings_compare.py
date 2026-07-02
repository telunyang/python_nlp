# 取得文本的嵌入向量，並比較兩句話的相似度

import asyncio
from ollama import AsyncClient
import time
import math


# 計算 cosine similarity
def cosine_similarity(vector1, vector2):
    dot_product = 0
    length1 = 0
    length2 = 0

    for a, b in zip(vector1, vector2):
        dot_product += a * b
        length1 += a * a
        length2 += b * b

    length1 = math.sqrt(length1)
    length2 = math.sqrt(length2)

    if length1 == 0 or length2 == 0:
        return 0

    return dot_product / (length1 * length2)

# 比較兩句話的相似度
async def compare_texts():
    # 定義兩句話
    text1 = '早安'
    text2 = '早上好'

    t1 = time.time()

    # 透過 embedding model 取得兩句話的嵌入向量
    client = await AsyncClient(
        host='http://localhost:11434',
        timeout=600,
    )
    response = await client.embed(
        model='bge-m3',
        input=[text1, text2],
        keep_alive='1h',
    )   

    # 取得第一句和第二句的向量
    embeddings = response.embeddings
    vector1 = embeddings[0]
    vector2 = embeddings[1]

    # 計算兩個向量的 cosine similarity
    similarity = cosine_similarity(vector1, vector2)

    print(f'文本1: {text1}')
    print(f'文本2: {text2}')
    print(f'語義相似度: {similarity:.4f}')

    t2 = time.time()
    print(f'執行時間: {t2 - t1:.2f} 秒')


if __name__ == "__main__":
    asyncio.run(compare_texts())