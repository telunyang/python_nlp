from flask import Flask, request, render_template
from openai import OpenAI
from modules.bm25 import load_qa
from sentence_transformers import SentenceTransformer
import faiss

# 讀取模型
model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
bi_encoder = SentenceTransformer(model_name)

# 讀取索引
index_path = './vector.index'
index = faiss.read_index(index_path)

'''
Flask 初始化
'''
app = Flask(__name__)
app.json.ensure_ascii = False # 防止中文變成 unicode 編碼

'''
OpenAI 設定初始化
'''
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

'''
變數初始化
'''
# 使用字典來保存每個會話的對話歷史
sessions_history = {}

'''
自訂 router
'''
# 首頁 (透過 render_template 函式，將 templates/index.html 檔案回傳給前端)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# 取得 ai assistant 的回應
@app.route("/chat", methods=["POST"])
def chat():
    # 取得前端傳來的 JSON 格式資料
    data = request.json

    # 取得 session id 和 使用者的訊息
    session_id = data.get("session_id")
    user_message = data.get("message")

    # 是否開啟檢索模型
    retrieved = True

    # 如果 session id 不存在於對話記錄中，針對這個 session id 建立一個新的會話記錄
    if session_id not in sessions_history:
        sessions_history[session_id] = [
            {"role": "system", "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}
        ]

    # 判斷是否進行檢索
    if retrieved:
        # 基本設定
        top_k = 3

        # 要查詢的字串
        query = user_message
        li_queries = [query]

        # 將查詢句子轉換成向量
        embeddings = bi_encoder.encode(
            li_queries, 
            batch_size=4, 
            show_progress_bar=False,
            normalize_embeddings=False
        )

        # 查詢
        D, I = index.search(embeddings, k=top_k)

        # 顯示結果
        list_scores = D.tolist()
        list_ids = I.tolist()
        print(f"相似度: {list_scores}")
        print(f"檢索的 Document IDs 為: {list_ids}")

        # 取得 QA 資料
        li_qa = load_qa()

        # 顯示前 top_k 高相似度的背景知識
        knowledge = ""
        sn = 1
        for doc_id in list_ids[0]: # [0] 是因為只有一個查詢句子
            knowledge += f"{sn}. {li_qa[doc_id]['BG']}\n"
        
        # 建立 user prompt
        user_message = f'''請參考以下資訊：
{knowledge}

==================================================

問題：
{user_message}

==================================================

答案是：'''

    # 將使用者的訊息，加入到會話記錄中
    sessions_history[session_id].append({"role": "user", "content": user_message})

    # 生成回應
    def generate_responses():
        # 透過 Chat Completions API 來取得 ai assistant 的回應
        completion = client.chat.completions.create(
            model="audreyt/Taiwan-LLM-7B-v2.0.1-chat-GGUF/Taiwan-LLM-7B-v2.0.1-chat-Q5_K_M.gguf",
            messages=sessions_history[session_id],
            temperature=0.7,
            stream=True,
        )

        # 透過 stream 方式，將 ai assistant 生成的文字一個一個輸出
        content = ''
        for chunk in completion:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content

        # 將生成的回應加入到會話歷史中
        sessions_history[session_id].append({"role": "assistant", "content": content})

    # 回傳 ai assistant 生成的回應
    return generate_responses(), {"Content-Type": "text/plain"}

if __name__ == "__main__":
    app.run(
        # 除錯模式為 True，服務執行期間有錯誤，會將 Traceback 顯示在網頁上，
        # 反之則顯示一般的 Internal Server Error
        debug=True,

        # 127.0.0.1 或 localhost 限定本機使用服務，
        # 0.0.0.0 代表所有知道主機實際 IP 的人都能存取 
        host='127.0.0.1',

        # 網址或 IP 後面附加的 Port 號，代表服務由該 Port 號提供
        port=5000 
    )
