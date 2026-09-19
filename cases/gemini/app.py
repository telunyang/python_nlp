import os
import json
from flask import Flask, request, Response, render_template
from google import genai
from google.genai.types import UserContent, ModelContent, Part
from dotenv import load_dotenv
load_dotenv(override=True)

# 建立 Flask app
app = Flask(__name__)

# 讀取環境變數
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemma-4-26b-a4b-it"
HISTORY_FILE = "chat_history.json"

# 建立 Gemini API client
client = genai.Client(api_key=GOOGLE_API_KEY)


# 讀取歷史對話
def load_history():
    # 如果歷史檔案不存在，回傳空列表
    if not os.path.exists(HISTORY_FILE):
        return []

    # 如果存在，讀取並回傳歷史對話
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# 儲存歷史對話
def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# 將 JSON 歷史轉成 Gemini 需要的格式
def build_gemini_history(history):
    # Gemini 的歷史格式是 UserContent 和 ModelContent 的列表，每個 Content 包含多個 Part
    gemini_history = []

    # 將 JSON 歷史轉換成 Gemini 的歷史格式
    for msg in history:
        # 根據角色建立對應的 Content，並將文字放在 Part 中
        if msg["role"] == "user":
            # 使用 UserContent 包裝使用者訊息，並將文字放在 Part 中
            gemini_history.append(
                UserContent(parts=[Part(text=msg["text"])])
            )
        elif msg["role"] == "model":
            # 使用 ModelContent 包裝模型回覆，並將文字放在 Part 中
            gemini_history.append(
                ModelContent(parts=[Part(text=msg["text"])])
            )

    return gemini_history


'''
Flask API 定義：
- GET /api/history: 讀取歷史對話
- POST /api/chat/stream: 接收使用者訊息，串流回覆，並在串流結束後儲存對話歷史
- POST /api/clear: 清除歷史對話
'''
# 首頁路由，提供簡單的前端介面
@app.route("/")
def index():
    return render_template("index.html")

# 取得歷史對話的 API 路由，回傳 JSON 格式的歷史對話列表
@app.route("/api/history")
def get_history():
    return load_history()

# 接收使用者訊息並串流回覆的 API 路由，
# 使用 POST 方法，接收 JSON 格式的訊息內容
@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    # 從請求中取得 JSON 資料，並從中提取使用者訊息
    data = request.get_json()
    user_message = data["message"]

    # 讀取舊歷史
    history = load_history()

    # 建立 Gemini chat
    chat = client.chats.create(
        model=MODEL_NAME,
        history=build_gemini_history(history)
    )

    # 3. 串流產生回答
    def generate():
        # 放置完整回覆的變數，串流過程中會不斷累積文字到這個變數中
        full_reply = ""

        # 呼叫 send_message_stream 方法，傳入使用者訊息，並取得串流回應
        response = chat.send_message_stream(user_message)

        # 迭代串流回應中的每個 chunk，將文字累積到 full_reply 中，並 yield 每個 chunk 的文字
        for chunk in response:
            # 如果 chunk 有文字內容
            if chunk.text:
                # 將 chunk 中的文字累積到 full_reply 變數中，方便儲存所有的回覆內容
                full_reply += chunk.text

                # yield 這段文字給前端，讓前端能夠即時顯示模型的回覆內容，而不需要等到整個回覆完成後才顯示
                yield chunk.text

        # 串流結束後，儲存本輪對話
        history.append({
            "role": "user",
            "text": user_message
        })

        history.append({
            "role": "model",
            "text": full_reply
        })

        # 將更新後的歷史對話儲存回檔案中，這樣下次讀取歷史對話時就能看到這次的對話內容
        save_history(history)

    # 回傳一個 Response 物件，將 generate 函式作為內容，並設定 mimetype 為 text/plain，這樣前端就能以串流的方式接收模型的回覆內容
    return Response(generate(), mimetype="text/plain")

# 清除歷史對話的 API 路由，使用 POST 方法，
# 當收到請求時會將歷史對話清空，並回傳一個 JSON 格式的狀態訊息
@app.route("/api/clear", methods=["POST"])
def clear_history():
    save_history([])
    return {"status": "ok"}


# 啟動 Flask app，設定 debug 模式為 True，並指定 host 和 port
if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )