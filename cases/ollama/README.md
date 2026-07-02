# Ollama

## 安裝 Ollama
- [Download Ollama](https://ollama.com/download/windows)

## 指令說明
- [CLI Reference](https://docs.ollama.com/cli)

## 使用 Ollama 下載模型：
  - [qwen3.5](https://ollama.com/library/qwen3.5)
    - `ollama pull qwen3.5:0.8b`
  - [BGE-M3](https://ollama.com/library/bge-m3)
    - `ollama pull bge-m3`

## 透過指令使用 ollama
```bash
# 查看模型支援功能
ollama show qwen3.5:0.8b

# 啟動對話
ollama run qwen3.5:0.8b

# 圖片描述（順序不能錯）
ollama run qwen3.5:0.8b --think=false "請描述照片的內容，以及裡面所寫的文字。" "D:\teach\restful_api_ajax\ollama\example01.jpg"
```

## 參考網站
- [Ollama - The easiest way to build with open models](https://ollama.com/)
- [Ollama Python Library](https://github.com/ollama/ollama-python)