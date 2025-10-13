# AutoGen
- [AutoGen - A framework for building AI agents and applications](https://microsoft.github.io/autogen/stable/)
- [Models](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/models.html)

## AgentChat
An web-based UI for prototyping with agents without writing code. Built on AgentChat.
```bash
pip install -U autogenstudio
autogenstudio ui --port 8080 --appdir ./myapp
```

## Installation

### Python 套件
```bash
pip install -U "autogen-agentchat" "autogen-ext[openai,ollama,anthropic]"
```

### ollama

#### 安裝
```bash
# ubuntu
curl -fsSL https://ollama.com/install.sh | sh

# windows
# 網址: https://ollama.com/download
```

#### 編輯 ollama.service
```bash
sudo vi /etc/systemd/system/ollama.service
```

#### 設定 ollama models 路徑
```
[Service]
Environment="OLLAMA_MODELS=/data2/users/darren/ollama_models"
```

#### 設定 keepalive
```
[Service]
Environment="OLLAMA_KEEP_ALIVE=-1"
```

#### 設定 Host IP 和 Port 
```
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
```

#### 重新啟動 ollama
```bash
# 重新載入系統服務
sudo systemctl daemon-reload
sudo systemctl restart ollama.service

# 檢查 ollama 狀態
sudo systemctl status ollama.service
```

### API
- [ollama/docs/api.md](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [generate-request-with-options](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-request-with-options)
- [modelfile.md#valid-parameters-and-values](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values)

#### 支援 function calling 的 ollama models 列表
- [https://ollama.com/search?c=tools](https://ollama.com/search?c=tools)
