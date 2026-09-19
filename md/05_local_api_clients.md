# 本機呼叫 Colab 上的 Ollama API

前提：

- Ollama 已在 Colab 的 `127.0.0.1:11434` 執行。
- ngrok Traffic Policy 已設定 Host rewrite。
- ngrok tunnel 已啟動。
- 已取得目前的 HTTPS endpoint。

假設：

```text
https://xxxx.ngrok-free.app
```

## 1. 用 `requests` 測試連線

```python
import requests

OLLAMA_HOST = "https://xxxx.ngrok-free.app"

try:
    response = requests.get(
        OLLAMA_HOST + "/api/tags",
        timeout=10,
    )

    print("Status code:", response.status_code)
    print("Response body:", response.text)

    if response.status_code == 200:
        print("Ollama connection successful.")
    else:
        print("Failed to connect to Ollama.")

except requests.exceptions.RequestException as error:
    print(f"Error connecting to Ollama: {error}")
```

注意：

```python
OLLAMA_HOST = "https://xxxx.ngrok-free.app"
```

不要寫：

```python
OLLAMA_HOST = "https://xxxx.ngrok-free.app:11434"
```

因為 ngrok public endpoint 已經提供 HTTPS port。

## 2. Ollama Native API

```python
import requests

OLLAMA_HOST = "https://xxxx.ngrok-free.app"

response = requests.post(
    OLLAMA_HOST + "/api/chat",
    json={
        "model": "gpt-oss:20b",
        "messages": [
            {
                "role": "user",
                "content": "Explain BM25."
            }
        ],
        "stream": False,
    },
    timeout=300,
)

response.raise_for_status()

data = response.json()

print(data["message"]["content"])
```

## 3. OpenAI-compatible API

Ollama 支援部分 OpenAI-compatible endpoint。

安裝：

```bash
pip install openai
```

Python：

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://xxxx.ngrok-free.app/v1/",
    api_key="ollama",
)

response = client.chat.completions.create(
    model="gpt-oss:20b",
    messages=[
        {
            "role": "user",
            "content": "Explain BM25."
        }
    ],
)

print(response.choices[0].message.content)
```

這裡的：

```python
api_key="ollama"
```

只是因為 OpenAI Python client 需要 `api_key` 參數；Ollama local API 本身不是用這個值做真正的 API key 驗證。

## 4. 403 時不要只印 status code

不建議只有：

```python
print(response.status_code)
```

應同時印：

```python
print(response.status_code)
print(response.text)
```

這樣才能分辨回應究竟來自：

- ngrok。
- Ollama。
- Traffic Policy。
- 其他 upstream error。

如果使用 Ollama + ngrok 時持續得到 403，請先查看：

```text
04_ollama_ngrok_integration.md
```

尤其確認：

```bash
ngrok http 11434 \
  --traffic-policy-file /content/ollama-ngrok.yaml
```

是否真的有使用 Traffic Policy。
