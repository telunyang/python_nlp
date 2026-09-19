# 在 Google Colab 使用 Ollama

## 1. 安裝 Ollama

在 Colab Terminal：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

確認：

```bash
ollama --version
```

## 2. Colab 安裝後通常不會自動啟動 Ollama server

在一般有 systemd 的 Linux 主機上，Ollama installer 可以建立 systemd service。

但 Google Colab Runtime 通常不是以正常 systemd service 管理方式執行，因此安裝完成後直接：

```bash
ollama list
```

可能看到：

```text
Error: could not connect to ollama server, run 'ollama serve' to start it
```

這不代表 Ollama 安裝失敗，只代表 server 還沒啟動。

## 3. 第一次啟動 Ollama

前景執行：

```bash
ollama serve
```

第一次啟動可能看到：

```text
Couldn't find '/root/.ollama/id_ed25519'. Generating new private key.
Your new public key is:
...
```

這是第一次初始化的正常行為。

如果使用前景模式，這個 Terminal 會被 `ollama serve` 佔用。

## 4. 建議在 Colab 背景執行

```bash
nohup ollama serve > /tmp/ollama.log 2>&1 &
```

確認 process：

```bash
pgrep -af "ollama serve"
```

等待 API ready：

```bash
until curl -sf http://127.0.0.1:11434/api/tags >/dev/null; do
  sleep 1
done
```

確認：

```bash
curl http://127.0.0.1:11434/api/tags
```

或：

```bash
ollama list
```

查看 log：

```bash
tail -f /tmp/ollama.log
```

## 5. 下載模型

例如：

```bash
ollama pull gpt-oss:20b
```

確認：

```bash
ollama list
```

## 6. 先在 Colab 內測試

在處理 ngrok 前，先確認 Ollama 自己可用：

```bash
curl http://127.0.0.1:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss:20b",
    "messages": [
      {
        "role": "user",
        "content": "Explain BM25 in one paragraph."
      }
    ],
    "stream": false
  }'
```

如果這一步失敗，先處理 Ollama 或模型，不要先查 ngrok。

## 7. 確認是否真的使用 GPU

另一個 Colab Terminal：

```bash
watch -n 1 nvidia-smi
```

然後執行 inference。

也可以：

```bash
ollama ps
```

模型推論時通常可以看到 GPU memory 與 GPU utilization 增加。

## 8. T4 VRAM 不足

可以考慮：

- 使用較小模型。
- 使用較低 bit 數量化版本。
- 降低 context length。
- 降低 concurrent requests。
- 關閉其他 GPU process。
- 將 Ollama parallel request 數量限制為 1。

例如：

```bash
pkill ollama

OLLAMA_NUM_PARALLEL=1 \
nohup ollama serve > /tmp/ollama.log 2>&1 &
```

## 9. 停止 Ollama

```bash
pkill ollama
```

Colab Runtime 被移除後，手動安裝的 Ollama 與下載模型通常都需要重新建立，除非另行保存到持久化儲存。
