# 測試 ollama 遠端連線
import requests

# 設定 Ollama API 的 URL
OLLAMA_HOST = "http://localhost:11434"
# OLLAMA_HOST = "https://112c-136-66-88-68.ngrok-free.app"
# OLLAMA_HOST = "https://{NGROK_URL}" # 如果使用 ngrok，請取消註解並替換 {NGROK_URL} 為實際的 ngrok URL

try:
    # 發送 GET 請求到 Ollama API
    response = requests.get(OLLAMA_HOST + "/api/tags", timeout=5)

    # 檢查回應狀態碼
    if response.status_code == 200:
        print("Ollama connection successful.")
    else:
        print(f"Failed to connect to Ollama. Status code: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"Error connecting to Ollama: {e}")
