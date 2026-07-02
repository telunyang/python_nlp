# 使用 QLora 來微調自訂多輪對話模型

## 分階段進行微調（可自行修正）
```
第一階段：
Qwen/Qwen2.5-0.5B-Instruct
no_thinking
max_length=512 或 768
確認資料流程、loss、推論都正常

第二階段：
Qwen/Qwen2.5-1.5B-Instruct 或 Qwen/Qwen3-0.6B
max_length=768 或 1024
加入更多 chat_history

第三階段：
若有明確 reasoning 資料，再做 Qwen3 thinking 版
completion 裡保留 <think>...</think>
推論端決定要顯示或隱藏 thinking
```

## 程式說明
- `prepare_chat_data.py`：把 chat_history.json 轉成 SFT 資料。
  - 這份程式同時支援 Gemini 與 Ollama 的 role=model 格式。它輸出 train.jsonl 與 valid.jsonl，每筆是一個多輪 prompt-completion 樣本。
  ```bash
  python prepare_chat_data.py \
    --history chat_history.json \
    --out_dir data/sft_no_thinking \
    --mode no_thinking

  python prepare_chat_data.py \
    --history chat_history.json \
    --out_dir data/sft_thinking \
    --mode thinking
  ```
- `train_qlora.py`：低 VRAM QLoRA + TensorBoard + Early Stopping。
  - 這版用 TRL SFTTrainer，不手動 tokenize，也不手刻 `[INST]、`、`<|start_header_id|>`。SFTTrainer 會依 tokenizer 的 chat template 處理 conversational dataset。TRL 文件也說 SFTTrainer 支援 PEFT adapter training，而且 LoRA/adapter 訓練通常使用較高 learning rate，例如約 1e-4。
  ```bash
  執行 non-thinking：
  python train_qlora.py \
    --model_id Qwen/Qwen2.5-0.5B-Instruct \
    --data_dir data/sft_no_thinking \
    --output_dir outputs/qwen25_05b_no_thinking \
    --max_length 768 \
    --grad_accum 16

  執行 thinking，建議先用 Qwen3：
  python train_qlora.py \
    --model_id Qwen/Qwen3-0.6B \
    --data_dir data/sft_thinking \
    --output_dir outputs/qwen3_06b_thinking \
    --max_length 1024 \
    --grad_accum 16
  ```
- 檢視收斂過程與 early stopping，可在訓練過程中使用指令：`tensorboard --logdir outputs/qwen25_05b_no_thinking/runs`。
- `plot_loss.py`：把 log 畫成圖，也可以讀 trainer_state.json。
- `infer_lora.py`：測試 adapter 推論。

## 建立 Ollama 可用的 Modelfile
檔案可能的內容：
```
FROM qwen2.5:0.5b
ADAPTER ./outputs/qwen25_05b_no_thinking

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048

SYSTEM """你是一個使用繁體中文回答的多輪對話助理。"""
```
建立：
```
ollama create my-qwen-chat -f Modelfile
ollama run my-qwen-chat
```