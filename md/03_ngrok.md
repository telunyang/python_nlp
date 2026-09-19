# 在 Google Colab 安裝與使用 ngrok

## 1. 安裝 ngrok

目前可使用 ngrok 的 Apt repository：

```bash
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
  | tee /etc/apt/sources.list.d/ngrok.list \
  && apt update \
  && apt install -y ngrok
```

Colab Terminal 通常已經是 root，因此一般不需要 `sudo`。

確認：

```bash
ngrok version
```

## 2. 設定 authtoken

ngrok agent session 需要帳號認證。

設定：

```bash
ngrok config add-authtoken "<YOUR_NGROK_AUTHTOKEN>"
```

檢查：

```bash
ngrok config check
```

如果沒有設定 token，可能出現：

```text
ERROR: authentication failed: This ngrok session is not authenticated.
ERROR: ERR_NGROK_4018
```

不要把真實 token commit 到 GitHub、公開 Notebook 或論文。

## 3. 最基本 HTTP tunnel

假設本機 service 位於：

```text
127.0.0.1:11434
```

可以：

```bash
ngrok http 11434
```

成功後會得到類似：

```text
https://xxxx.ngrok-free.app
```

外部 client 不需要再加 `:11434`。

正確：

```text
https://xxxx.ngrok-free.app/api/tags
```

錯誤：

```text
https://xxxx.ngrok-free.app:11434/api/tags
```

## 4. `--host-header` 已 deprecated

舊寫法：

```bash
ngrok http 11434 --host-header="localhost:11434"
```

新版 ngrok 會提示：

```text
Flag --host-header has been deprecated, use traffic policy instead
```

所以不要再把它當成目前的主要用法。

如果需要改寫 `Host` header，改用 Traffic Policy。

## 5. Traffic Policy 基本形式

例如：

```yaml
on_http_request:
  - actions:
      - type: add-headers
        config:
          headers:
            host: localhost
```

然後：

```bash
ngrok http 11434 --traffic-policy-file /content/ollama-ngrok.yaml
```

Ollama 的完整情境請看：

```text
04_ollama_ngrok_integration.md
```

## 6. 常見錯誤

### `ERR_NGROK_4018`

原因：

```text
ngrok 尚未設定有效 authtoken
```

處理：

```bash
ngrok config add-authtoken "<YOUR_NGROK_AUTHTOKEN>"
ngrok config check
```

### Tunnel 可以啟動，但 upstream 回 403

這代表 HTTPS tunnel 本身不一定有問題。

如果 upstream 是 Ollama，可能是 `Host` header 不符合 Ollama 預期。請使用 Traffic Policy 將：

```text
Host
```

改為：

```text
localhost
```

詳細設定請看 `04_ollama_ngrok_integration.md`。

## 7. 停止 ngrok

如果在前景：

```text
Ctrl + C
```

如果在背景：

```bash
pkill ngrok
```

ngrok endpoint 與 Colab Runtime 都不是永久服務，重新建立環境後應重新確認目前 endpoint。
