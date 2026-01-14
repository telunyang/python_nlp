'''
匯入套件
'''
import os
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

'''
[faiss.IndexFlatL2]
使用歐氏距離

[faiss.IndexFlatIP]
IP = Inner Product，
測試使用 Inner Product 來比較 features 資料
同時進行 feature normalization
等同於 cosine similarity
'''

# 模型名稱
model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'

# 索引存放路徑
index_path = './vector.index'

# 讀取 model
bi_encoder = SentenceTransformer(model_name)

# 讀取問答資料：「句子」與對應的「句子 ID」(需要 int)
docs = []
doc_ids = []
with open('../lm_studio/qa.json', 'r', encoding='utf-8') as f:
    # 讀取 json 檔案
    li_qa = json.loads(f.read())

    # 準備文件與文件 ID
    for index, qa in enumerate(li_qa):
        docs.append(qa['Q'])
        doc_ids.append(index)

    # 將 doc_ids 轉成 numpy array，並指定資料型態為 int64
    doc_ids = np.array(doc_ids).astype('int64')

# 將所有句子轉換成向量，同時計算轉向量時間
embeddings = bi_encoder.encode(
    docs, 
    batch_size=8,
    show_progress_bar=True,
    normalize_embeddings=False # 建議先查詢預訓練模型是否支援
)

# 索引預設變數
index = None

# 讀取索引，不存在就初始化
if not os.path.exists(index_path):
    dims = embeddings.shape[1]
    index = faiss.IndexFlatIP(dims) # 初始化索引的維度
    index = faiss.IndexIDMap(index) # 讓 index 有記錄對應 doc id 的能力
else:
    # 索引存在，直接讀取
    index = faiss.read_index(index_path)

# 加入 doc id 到 對應的 vector
print(embeddings.shape)
print(len(doc_ids))
index.add_with_ids(embeddings, doc_ids) # 加入 向量 與 文件ID
# index.add(embeddings) # 僅加入向量

# 儲存索引
faiss.write_index(index, index_path)