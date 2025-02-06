'''
匯入套件
'''
from sentence_transformers import SentenceTransformer
import faiss
import json

# 基本設定
model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
bi_encoder = SentenceTransformer(model_name)

# 讀取索引
index_path = './vector.index'
index = faiss.read_index(index_path)

# 查詢句子
list_query = ['為什麼雪是白色的', '為什麼太陽會升起和落下', '為什麼我們有五官']

# 將查詢句子轉換成向量
embeddings = bi_encoder.encode(
    list_query, 
    batch_size=3, 
    show_progress_bar=False,
    normalize_embeddings=False
)

# 查詢
D, I = index.search(embeddings, k=3)

# 顯示結果
list_scores = D.tolist()
list_ids = I.tolist()
print(f"相似度: {list_scores}")
print(f"檢索的 Document IDs 為: {list_ids}")

with open('../lm_studio/qa.json', 'r', encoding='utf-8') as f:
    li_qa = json.loads(f.read())

for index, li_ids in enumerate(list_ids):
    print(f"查詢問題: {list_query[index]}")
    for id in li_ids:
        print(f"- 相似問題: {li_qa[id]['Q']}，Document ID: {i}，相似度: {list_scores[ index ][ li_ids.index(id) ]}")