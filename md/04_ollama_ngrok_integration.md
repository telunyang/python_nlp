# Ollama + ngrok 整合

## 1. 架構

```text
本機 Python / curl / OpenAI SDK
        │
        │ HTTPS
        ▼
ngrok public endpoint
        │
        │ Traffic Policy
        │ Host → localhost
        ▼
Colab 127.0.0.1:11434
        │
        ▼
Ollama
        │
        ▼
模型 / GPU
```

## 2. 前置條件

在 Colab 先確認 Ollama 正常：

```bash
curl http://127.0.0.1:11434/api/tags
```

必須先得到正常 JSON。

再確認 ngrok：

```bash
ngrok config check
```

## 3. 為什麼直接 `ngrok http 11434` 可能得到 403

如果直接：

```bash
ngrok http 11434
```

外部 client 傳進來的 HTTP request 可能帶有：

```text
Host: xxxx.ngrok-free.app
```

而 Ollama 可能拒絕這個 Host，最後 client 只看到：

```text
HTTP 403 Forbidden
```

這種情況下：

- ngrok tunnel 其實已經連通。
- Ollama server 也可能正在執行。
- 問題發生在轉送 request 的 Host header。

## 4. 建立 Ollama 專用 Traffic Policy

在 Colab 建立：

```bash
cat > /content/ollama-ngrok.yaml <<'EOF'
on_http_request:
  - actions:
      - type: add-headers
        config:
          headers:
            host: localhost
EOF
```

檢查：

```bash
cat /content/ollama-ngrok.yaml
```

內容應為：

```yaml
on_http_request:
  - actions:
      - type: add-headers
        config:
          headers:
            host: localhost
```

## 5. 使用 Traffic Policy 啟動 ngrok

先停止舊的 ngrok session。

如果在前景：

```text
Ctrl + C
```

接著：

```bash
ngrok http 11434 \
  --traffic-policy-file /content/ollama-ngrok.yaml
```

這是目前整合 Ollama 時的重要寫法。

不要再使用：

```bash
--host-header="localhost:11434"
```

因為新版 ngrok 已將它標為 deprecated。

## 6. 記下 ngrok HTTPS endpoint

例如：

```text
https://xxxx.ngrok-free.app
```

後續稱為：

```text
NGROK_URL
```

不要額外加 `:11434`。

## 7. 從本機測試 `/api/tags`

Python：

```python
import requests

OLLAMA_HOST = "https://xxxx.ngrok-free.app"

response = requests.get(
    OLLAMA_HOST + "/api/tags",
    timeout=10,
)

print("Status:", response.status_code)
print("Body:", response.text)

response.raise_for_status()
```

如果成功，應為：

```text
HTTP 200
```

並看到 Ollama model list。

## 8. 403 排查

如果：

```python
response.status_code == 403
```

先印出：

```python
print(response.text)
```

接著依序確認。

### 8.1 Ollama 本機 API

Colab：

```bash
curl -i http://127.0.0.1:11434/api/tags
```

應為：

```text
HTTP/1.1 200 OK
```

### 8.2 模擬外部 Host

可以測：

```bash
curl -i \
  -H "Host: example.ngrok-free.app" \
  http://127.0.0.1:11434/api/tags
```

如果這裡出現 403，而普通 localhost request 是 200，就很可能是 Host header 問題。

### 8.3 確認 Traffic Policy 有載入

ngrok 必須用：

```bash
ngrok http 11434 \
  --traffic-policy-file /content/ollama-ngrok.yaml
```

而不是只有：

```bash
ngrok http 11434
```

## 9. 安全性

Ollama local API 預設不是為公開 Internet service 設計的。

ngrok URL 一旦暴露，任何取得 URL 的人都可能對 endpoint 發 request。

至少應：

- 不公開 ngrok URL。
- 不把 URL commit 到 public repository。
- 測試完成後停止 ngrok。
- 需要較正式的存取控制時，使用 ngrok Traffic Policy 加 authentication。

## 10. 最短整合流程

```bash
# 1. Ollama
nohup ollama serve > /tmp/ollama.log 2>&1 &

until curl -sf http://127.0.0.1:11434/api/tags >/dev/null; do
  sleep 1
done

# 2. Traffic Policy
cat > /content/ollama-ngrok.yaml <<'EOF'
on_http_request:
  - actions:
      - type: add-headers
        config:
          headers:
            host: localhost
EOF

# 3. ngrok
ngrok config check

ngrok http 11434 \
  --traffic-policy-file /content/ollama-ngrok.yaml
```
