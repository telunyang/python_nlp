# 繁體中文多輪對話 QLoRA

這是一份刻意簡化、以教學為導向的繁體中文多輪對話微調專案。

專案使用：

- Hugging Face Datasets
- Transformers
- PEFT / LoRA
- TRL `SFTTrainer`
- bitsandbytes 4-bit NF4 QLoRA
- PyTorch
- Matplotlib

預設模型：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

預設資料集：

```text
renhehuang/coffee-order-zhtw
```

整個專案不使用額外的 `config.json`、YAML 或其他設定檔。

所有主要參數都直接放在各個 `.py` 檔案上方，方便初學者直接閱讀、修改與理解。

---

## 1. 專案目標

這個專案示範一個完整的多輪對話 QLoRA 流程：

```text
Hugging Face Dataset
        ↓
download_chat_data.py
        ↓
data/chat.jsonl
        ↓
prepare_chat_data.py
        ↓
data/sft/train.jsonl
data/sft/valid.jsonl
        ↓
train_qlora.py
        ↓
4-bit QLoRA Fine-tuning
        ↓
outputs/qwen25_05b_lora/
        ↓
plot_loss.py
        ↓
loss_curve.png

以及：

outputs/qwen25_05b_lora/
        ↓
infer_lora.py
        ↓
單輪推論

或：

outputs/qwen25_05b_lora/
        ↓
infer_lora_chat.py
        ↓
多輪互動式對話
```

---

## 2. 建議執行環境

這個專案預設以 NVIDIA GPU 為主。

目前範例主要針對：

```text
Google Colab
NVIDIA Tesla T4
約 15 GB 可用 GPU RAM
```

進行設定。

`train_qlora.py` 目前採用：

```text
Base model weights
    → 4-bit NF4

4-bit compute dtype
    → FP16

LoRA trainable parameters
    → FP32

Trainer mixed precision
    → FP16
```

Tesla T4 不使用 BF16 mixed precision。

---

## 3. 安裝套件

先安裝專案所需套件：

```bash
pip install -r requirements-colab.txt
```

如果目前 Colab 環境中有舊版且不相容的 `torchao`，而 PEFT 出現：

```text
ImportError:
Found an incompatible version of torchao.
Found version 0.10.0, but only versions above 0.16.0 are supported
```

本專案並沒有使用 TorchAO quantization，而是使用 bitsandbytes NF4，因此可以移除：

```bash
python -m pip uninstall -y torchao
```

---

## 4. 建議執行順序

完整流程：

```bash
pip install -r requirements-colab.txt

python download_chat_data.py

python prepare_chat_data.py

python train_qlora.py

python plot_loss.py

python infer_lora.py

python infer_lora_chat.py
```

其中：

```text
download_chat_data.py
```

下載並整理原始多輪對話資料。

```text
prepare_chat_data.py
```

建立真正給 TRL SFT 使用的 train / validation JSONL。

```text
train_qlora.py
```

執行 QLoRA 微調。

```text
plot_loss.py
```

輸出 training / validation loss 圖。

```text
infer_lora.py
```

進行單輪推論。

```text
infer_lora_chat.py
```

進行多輪互動式對話。

---

## 5. 建議的專案目錄

```text
llm_finetune/
├── download_chat_data.py
├── prepare_chat_data.py
├── train_qlora.py
├── plot_loss.py
├── infer_lora.py
├── infer_lora_chat.py
├── requirements-colab.txt
├── README.md
│
├── data/
│   ├── chat.jsonl
│   └── sft/
│       ├── train.jsonl
│       └── valid.jsonl
│
└── outputs/
    └── qwen25_05b_lora/
        ├── checkpoint-xxx/
        ├── adapter_config.json
        ├── adapter_model.safetensors
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── trainer_state.json
        └── loss_curve.png
```

實際檔案內容會依 Transformers、PEFT 與 tokenizer 版本略有不同。

---

# 6. 資料集

預設使用公開、不需要 gated access 的繁體中文多輪對話資料：

```text
renhehuang/coffee-order-zhtw
```

資料內容主要是咖啡點餐多輪對話，任務範圍單純，適合用來示範：

- system prompt
- user / assistant roles
- multi-turn context
- LoRA / QLoRA
- supervised fine-tuning
- validation loss
- checkpoint
- early stopping
- multi-turn inference

---

## 7. 下載原始資料

執行：

```bash
python download_chat_data.py
```

程式會：

```text
Hugging Face Dataset
        ↓
讀取 train split
        ↓
shuffle
        ↓
過濾太短的 conversation
        ↓
最多保留 MAX_CONVERSATIONS
        ↓
輸出 data/chat.jsonl
```

預設：

```python
MAX_CONVERSATIONS = 2500
MIN_MESSAGES = 4
SEED = 42
```

如果只是想快速測試整個 pipeline，可以把：

```python
MAX_CONVERSATIONS = 2500
```

改成：

```python
MAX_CONVERSATIONS = 500
```

甚至：

```python
MAX_CONVERSATIONS = 100
```

先確認流程可以完整跑通。

---

## 8. `data/chat.jsonl` 格式

`download_chat_data.py` 輸出的每一行都是一整段 conversation。

例如：

```json
{"messages":[
  {"role":"user","content":"我想點一杯冰拿鐵"},
  {"role":"assistant","content":"好的，一杯大杯冰拿鐵。"},
  {"role":"user","content":"幫我加一份濃縮"},
  {"role":"assistant","content":"好的，已幫您加一份濃縮咖啡。"}
]}
```

這個階段仍然保留完整 conversation，尚未拆成 SFT examples。

---

# 9. 準備 SFT 資料

執行：

```bash
python prepare_chat_data.py
```

這支程式最重要的設計是：

> 先以完整 conversation 切 train / validation，再展開成 prompt / completion。

流程：

```text
完整 conversations
        ↓
固定 seed shuffle
        ↓
以 conversation 為單位切分
        ↓
90% train
10% validation
        ↓
各自展開成 prompt / completion
        ↓
data/sft/train.jsonl
data/sft/valid.jsonl
```

---

## 10. 為什麼不能先展開再切 train / validation？

假設一段原始 conversation：

```text
user: 我要冰拿鐵
assistant: 好的
user: 加一份濃縮
assistant: 沒問題
```

如果先展開：

```text
Example A
prompt:
    user: 我要冰拿鐵
completion:
    assistant: 好的
```

以及：

```text
Example B
prompt:
    user: 我要冰拿鐵
    assistant: 好的
    user: 加一份濃縮
completion:
    assistant: 沒問題
```

如果 A 被分到 train，而 B 被分到 validation，

validation 其實已經包含模型在 train 看過的同一段對話內容。

這會造成：

```text
turn-level leakage
```

因此本專案固定採用：

```text
Conversation Split First
→ Example Expansion Second
```

---

## 11. SFT 的 prompt / completion 格式

例如：

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "我想點一杯冰拿鐵"
    },
    {
      "role": "assistant",
      "content": "好的，一杯大杯冰拿鐵。"
    },
    {
      "role": "user",
      "content": "幫我加一份濃縮"
    }
  ],
  "completion": [
    {
      "role": "assistant",
      "content": "好的，已幫您加一份濃縮咖啡。"
    }
  ]
}
```

可以理解成：

```text
prompt
    =
模型回答之前可以看到的 conversation history

completion
    =
這一筆 training example 希望模型學會產生的 assistant response
```

目前：

```python
MAX_CONTEXT_MESSAGES = 6
```

表示 prompt 最多保留最近 6 則非 system messages。

大約相當於最近 3 輪：

```text
user
assistant
user
assistant
user
assistant
```

如果原始 conversation 有 system message，system 會固定保留。

---

# 12. 開始 QLoRA 訓練

執行：

```bash
python train_qlora.py
```

如果想同時統計總執行時間：

```bash
time python train_qlora.py
```

預設模型：

```python
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
```

---

## 13. QLoRA 的基本概念

QLoRA 可以簡單拆成：

```text
Q
=
Quantization

LoRA
=
Low-Rank Adaptation
```

本專案的 base model 使用：

```text
4-bit NF4
```

降低 GPU RAM 使用量。

原始 base model 大部分參數：

```text
requires_grad=False
```

也就是不直接更新。

真正訓練的是另外加入的 LoRA adapter：

```text
requires_grad=True
```

因此即使原始模型接近 5 億參數，真正更新的參數只有約數百萬個。

---

## 14. 目前 T4 訓練設定

目前主要設定：

```python
MAX_LENGTH = 768

EPOCHS = 5

LEARNING_RATE = 1e-4

PER_DEVICE_TRAIN_BATCH_SIZE = 8

PER_DEVICE_EVAL_BATCH_SIZE = 16

GRADIENT_ACCUMULATION_STEPS = 2

GRADIENT_CHECKPOINTING = False
```

因此 effective training batch size 約為：

```text
8 × 2 = 16
```

也就是：

```text
micro-batch size
    =
8

gradient accumulation
    =
2

effective batch size
    =
16
```

這樣比：

```text
batch=1
gradient accumulation=16
```

更能利用 Tesla T4 的平行運算能力。

---

## 15. 如果發生 CUDA Out of Memory

目前設定是為了提高 T4 GPU 利用率。

如果發生：

```text
CUDA out of memory
```

可以先把：

```python
GRADIENT_CHECKPOINTING = False
```

改成：

```python
GRADIENT_CHECKPOINTING = True
```

如果仍然 OOM，再退回：

```python
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
```

effective batch size仍然是：

```text
4 × 4 = 16
```

如果還需要更保守：

```python
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
```

仍然：

```text
2 × 8 = 16
```

---

## 16. T4 上的 FP16 / BF16

本專案針對 Tesla T4 固定使用：

```python
bnb_4bit_compute_dtype=torch.float16
```

以及：

```python
fp16=True
bf16=False
tf32=False
```

LoRA trainable parameters 則保持：

```text
FP32
```

因此整體 precision 結構是：

```text
Base model storage
    → 4-bit NF4

Matrix compute
    → FP16

Trainer mixed precision
    → FP16

LoRA trainable parameters
    → FP32
```

這是為了避免 T4 上的 AMP / BF16 gradient 相容性問題。

---

# 17. Checkpoint 與續訓

`train_qlora.py` 預設每 50 個 optimizer steps：

1. 做一次 validation。
2. 儲存一份 `checkpoint-xxx`。
3. 檢查 validation loss 是否改善。
4. 更新 early stopping 狀態。

主要設定：

```python
CHECK_STRATEGY = "steps"

CHECK_STEPS = 50

CHECKPOINT_LIMIT = 3

EARLY_STOPPING_PATIENCE = 3

RESUME_FROM_CHECKPOINT = None
```

---

## 18. `RESUME_FROM_CHECKPOINT` 三種用法

### 重新開始

```python
RESUME_FROM_CHECKPOINT = None
```

表示：

```text
不要讀取舊 checkpoint
→ 從頭開始一個新的訓練
```

---

### 自動找最新 checkpoint

```python
RESUME_FROM_CHECKPOINT = True
```

Trainer 會嘗試從 `OUTPUT_DIR` 中最新 checkpoint 繼續。

---

### 指定 checkpoint

```python
RESUME_FROM_CHECKPOINT = "outputs/qwen25_05b_lora/checkpoint-200"
```

表示直接從：

```text
checkpoint-200
```

恢復訓練。

---

## 19. Checkpoint 會恢復什麼？

完整 Trainer checkpoint 通常不只包含 LoRA adapter。

續訓時還需要恢復：

```text
LoRA weights
optimizer state
learning-rate scheduler state
global training step
Trainer state
callback state
```

本專案使用：

```python
restore_callback_states_from_checkpoint=True
```

所以 early stopping 已經累積的 patience 狀態也可以跟著恢復。

---

# 20. Early stopping

預設：

```python
EARLY_STOPPING_PATIENCE = 3
```

表示：

如果連續 3 次 validation：

```text
eval_loss
```

都沒有比目前最佳值更低，就提前停止訓練。

例如：

```text
Eval 1
eval_loss = 1.20

Eval 2
eval_loss = 1.10
→ 改善

Eval 3
eval_loss = 1.12
→ 沒改善，第 1 次

Eval 4
eval_loss = 1.15
→ 沒改善，第 2 次

Eval 5
eval_loss = 1.14
→ 沒改善，第 3 次
```

這時就可能觸發 early stopping。

---

# 21. Best model 存在哪裡？

訓練設定：

```python
load_best_model_at_end=True

metric_for_best_model="eval_loss"

greater_is_better=False
```

表示：

```text
eval_loss 越低越好
```

Trainer 會記錄哪一個：

```text
OUTPUT_DIR/checkpoint-xxx
```

具有最低 validation loss。

訓練結束或 early stopping 後，

最佳 checkpoint 會自動重新載回記憶體。

接著程式執行：

```python
trainer.save_model(OUTPUT_DIR)
```

因此：

```text
outputs/qwen25_05b_lora/
```

根目錄本身就是最後可以直接拿來推論的最佳 LoRA adapter。

例如可能包含：

```text
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
trainer_state.json
```

---

## 22. `CHECKPOINT_LIMIT = 3`

目前：

```python
CHECKPOINT_LIMIT = 3
```

不是單純表示：

> 只保留最新 3 份。

當：

```python
load_best_model_at_end=True
```

時，Trainer 會考慮保留最佳 checkpoint。

因此設定 3 的目的主要是：

```text
保留最佳模型 checkpoint
+
保留近期可續訓 checkpoint
+
限制磁碟使用量
```

---

# 23. 每個 epoch 才 evaluation / checkpoint

如果不想每 50 steps 做一次：

```python
CHECK_STRATEGY = "steps"
```

可以改成：

```python
CHECK_STRATEGY = "epoch"
```

此時：

```text
每個 epoch 結束
    ↓
evaluation
    ↓
save checkpoint
    ↓
early stopping check
```

`CHECK_STEPS` 就不再影響 evaluation / checkpoint 的頻率。

---

# 24. 畫 Training / Validation Loss

訓練完成後：

```bash
python plot_loss.py
```

程式會讀：

```text
outputs/qwen25_05b_lora/trainer_state.json
```

其中的：

```text
log_history
```

並取出：

```text
loss
eval_loss
```

最後輸出：

```text
outputs/qwen25_05b_lora/loss_curve.png
```

---

## 25. 為什麼不用 `plt.show()`？

如果是在：

```text
VS Code
+
Google Colab Extension
+
Colab Remote Terminal
```

執行 `.py`，

這個 Linux remote environment 通常沒有一般桌面 GUI。

因此：

```python
plt.show()
```

不一定會彈出視窗。

目前 `plot_loss.py` 使用：

```python
matplotlib.use("Agg")
```

以及：

```python
plt.savefig(...)
```

直接將圖表輸出成 PNG。

---

## 26. 如何觀察 overfitting？

理想狀況：

```text
Training Loss
    ↓

Validation Loss
    ↓
```

如果後期出現：

```text
Training Loss
    持續下降

Validation Loss
    開始上升
```

通常代表模型可能逐漸 overfit training data。

這也是使用：

```text
validation
+
best checkpoint
+
early stopping
```

的原因。

---

# 27. 單輪推論

執行：

```bash
python infer_lora.py
```

程式會從：

```text
outputs/qwen25_05b_lora
```

載入：

```text
Qwen2.5 Base Model
+
最佳 LoRA Adapter
+
Tokenizer
```

然後測試一筆：

```text
system
+
user
→
assistant
```

---

## 28. 測試特定 checkpoint

如果要測試：

```text
checkpoint-200
```

可以把 `infer_lora.py` 中：

```python
ADAPTER_DIR = "outputs/qwen25_05b_lora"
```

改成：

```python
ADAPTER_DIR = "outputs/qwen25_05b_lora/checkpoint-200"
```

這樣可以比較：

```text
checkpoint-50
checkpoint-100
checkpoint-200
best adapter
```

之間的實際回答差異。

---

# 29. 多輪互動式推論

執行：

```bash
python infer_lora_chat.py
```

啟動後可以直接在 terminal 對話。

例如：

```text
你：我想點一杯冰拿鐵。

助理：好的，一杯大杯冰拿鐵。需要幫您加一份濃縮嗎？

你：好，幫我加。

助理：好的，已幫您的大杯冰拿鐵加一份濃縮咖啡。

你：再來一杯熱的。

助理：好的，再幫您加一杯大杯熱拿鐵。
```

第二輪：

```text
好，幫我加
```

沒有重新說：

```text
冰拿鐵
```

但模型仍然可以使用前面的 conversation history 理解上下文。

---

## 30. 多輪對話指令

`infer_lora_chat.py` 支援：

```text
/reset
```

清除目前 user / assistant history，重新開始一段新 conversation。

```text
/history
```

顯示目前保存的 conversation history。

```text
/help
```

顯示指令。

以下都可以結束：

```text
exit
quit
q
```

---

## 31. 多輪 history 限制

預設：

```python
MAX_HISTORY_TURNS = 10
```

最多保留最近 10 組：

```text
user
assistant
```

另外 system prompt 會固定保留。

限制 history 的原因是：

```text
conversation 越長
    ↓
token 數越多
    ↓
context 越長
    ↓
GPU RAM 使用量增加
    ↓
推論速度下降
```

因此不建議讓 terminal conversation 無限制增長。

---

# 32. Colab 的 `/content`

如果專案放在：

```text
/content/llm_finetune
```

請注意：

```text
/content
```

屬於 Colab Runtime 的暫時儲存空間。

Runtime 被刪除後，裡面的：

```text
data/
outputs/
checkpoint/
adapter/
loss_curve.png
```

都有可能一起消失。

所以訓練完成後應先備份重要輸出。

---

## 33. 將 output 下載到本機

如果使用：

```text
VS Code
+
Google Colab Extension
```

可以先在 Colab terminal 把 output 資料夾打包。

例如：

```bash
cd /content/llm_finetune

tar -czf qwen25_05b_lora.tar.gz \
    outputs/qwen25_05b_lora
```

接著在 VS Code：

```text
Colab
→ Contents
→ Refresh
→ qwen25_05b_lora.tar.gz
→ 右鍵
→ Download...
```

即可下載到本機。

---

## 34. 只想保留最後最佳 LoRA adapter

如果不需要續訓，

通常不一定需要保存所有：

```text
checkpoint-50
checkpoint-100
checkpoint-150
...
```

可以只保留：

```text
outputs/qwen25_05b_lora/
```

根目錄中的最佳 LoRA adapter 與 tokenizer。

但如果未來要：

```python
trainer.train(
    resume_from_checkpoint=...
)
```

則 checkpoint 仍然需要保留。

---

# 35. 檔案用途

## `download_chat_data.py`

下載公開繁體中文多輪對話資料集。

主要工作：

```text
Hugging Face Dataset
→ shuffle
→ 過濾短 conversation
→ 限制資料量
→ data/chat.jsonl
```

---

## `prepare_chat_data.py`

將完整 conversation：

```text
先切 train / validation
```

再展開成：

```text
prompt
→
completion
```

避免同一段 conversation 同時出現在 train 與 validation。

輸出：

```text
data/sft/train.jsonl
data/sft/valid.jsonl
```

---

## `train_qlora.py`

執行：

```text
4-bit NF4
+
LoRA
+
FP16 mixed precision
+
checkpoint
+
resume training
+
validation
+
best model
+
early stopping
```

並儲存最後最佳 LoRA adapter。

---

## `plot_loss.py`

讀取：

```text
trainer_state.json
```

繪製：

```text
Training Loss
+
Validation Loss
```

輸出：

```text
loss_curve.png
```

---

## `infer_lora.py`

載入最佳 LoRA adapter，進行單輪測試推論。

---

## `infer_lora_chat.py`

載入最佳 LoRA adapter，啟動互動式多輪對話。

支援：

```text
/reset
/history
/help
exit
```

---

# 36. 最短操作版

如果環境已經安裝好 dependency：

```bash
python download_chat_data.py

python prepare_chat_data.py

time python train_qlora.py

python plot_loss.py

python infer_lora.py

python infer_lora_chat.py
```

整體資料流：

```text
Hugging Face Dataset
        ↓
data/chat.jsonl
        ↓
data/sft/train.jsonl
data/sft/valid.jsonl
        ↓
Qwen2.5-0.5B-Instruct
        +
4-bit NF4
        +
LoRA
        ↓
Best LoRA Adapter
        ↓
Loss Plot
        +
Single-turn Inference
        +
Multi-turn Chat
```
