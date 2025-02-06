import re
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, OkapiBM25Model
from gensim.similarities import SparseMatrixSimilarity
import jieba
import heapq
import json

# 讀取問答資料
def load_qa():
    with open('./qa.json', 'r', encoding='utf-8') as f:
        li_qa = json.loads(f.read())
    return li_qa

# 取得 QA 當中的 Q
def get_questions(li_qa):
    li_questions = []
    for qa in li_qa:
        li_questions.append(qa['Q'])
    return li_questions

# 將文件進行分詞/斷詞
def tokenize(li_docs):
    li_seq = []
    for seq in li_docs:
        seq = re.sub(r"\s| |\n", "", seq) # 簡單清理
        li_seq.append(jieba.lcut(seq))
    return li_seq
    

# 用 BM25 模型計算相似度(相關性)
'''
定義函式 bm25_similarity：
- list_seq 是一個文件的列表，每個文件都是一個詞的列表；
- query 是查詢字符串。
'''
def bm25_similarity(list_seq, query):
    '''
    建立詞典：
    - 使用 list_seq 中的所有詞彙建立一個 Dictionary 物件，為每個詞彙分配一個唯一的ID。

    Dictionary 用法，可參考: https://blog.csdn.net/Don_t_always_ail/article/details/122239023
    '''
    dictionary = Dictionary(list_seq)

    '''
    初始化 BM25 模型：
    - 使用先前建立的詞典，來初始化一個 OkapiBM25Model 物件，這是用於資訊檢索 (Information Retrieval) 的統計模型。
    '''
    bm25_model = OkapiBM25Model(dictionary=dictionary)

    '''
    建立 BM25 語料庫：
    - dictionary.doc2bow：將每個文件（詞的列表）轉換為詞袋（Bag-of-Words）表示，即每個詞的ID及其在文件中的出現次數。
    - map 函式：對 list_seq 中的每個文件應用 doc2bow 轉換。
    - bm25_model[...]：對轉換後的語料庫應用 BM25 權重，得到加權的語料庫 bm25_corpus。
    '''
    bm25_corpus = bm25_model[list(map(dictionary.doc2bow, list_seq))]

    '''
    SparseMatrixSimilarity：創建一個稀疏矩陣相似度對象，用於快速計算查詢與文件之間的相似度。
    - bm25_corpus：BM25 加權的語料庫。
    - num_docs：文件數量。
    - num_terms：詞典中詞的數量。
    - normalize_queries=False：不對查詢進行 normalization。
    - normalize_documents=False：不對文件進行 normalization。
    '''
    bm25_index = SparseMatrixSimilarity(
        bm25_corpus, 
        num_docs=len(list_seq), 
        num_terms=len(dictionary), 
        normalize_queries=False, 
        normalize_documents=False
    )

    '''
    對查詢進行分詞：
    - 使用 jieba.lcut 對查詢字符串進行分詞。
    '''
    query = jieba.lcut(query)

    '''
    建立 TF-IDF 模型：
    - 參數 smartirs='bnn'：指定權重計算的方式，其中 'bnn' 表示二值化的詞頻和未正則化的逆向文件頻率 (IDF)。
    '''
    tfidf_model = TfidfModel(
        dictionary=dictionary, 
        smartirs='bnn' # default: nfc
    )

    '''
    轉換查詢向量：
    - dictionary.doc2bow(query)：將查詢詞列表轉換為詞袋表示。
    - tfidf_model[...]：對詞袋表示的查詢應用 TF-IDF 權重，得到加權的查詢向量 tfidf_query。
    '''
    tfidf_query = tfidf_model[dictionary.doc2bow(query)]

    '''
    計算相似度：
    - 使用之前建立的 bm25_index，計算查詢向量與語料庫中每個文件的相似度。
    '''
    similarities = bm25_index[tfidf_query]
    
    return similarities


# 顯示前 top_k 高相似度的文章
def get_page_score(similarities, top_k):
    '''
    (文章索引, 查詢後的相似分數)
    [(0, 0.0), (1, 0.0), (2, 0.0), (3, 3.4350295), (4, 2.4153714), (5, 0.0), (6, 0.0), ...]
    '''
    similarities = list(enumerate(similarities))

    '''
    前 top_k 高相似度的文章
    [(176, 15.524609), (177, 9.0146), (116, 7.6217546), (100, 6.096363), (159, 4.855205)]
    '''
    top_items  = heapq.nlargest(
        top_k, 
        similarities,
        key=lambda x: x[1]
    )

    return [(doc_index, score) for doc_index, score in top_items]