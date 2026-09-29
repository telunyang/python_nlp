# Laya 繁體中文咖啡點餐教學版

這個專案使用臺灣繁體中文咖啡點餐資料，將對話轉成 Laya 的 `choice`、`noul`、`score` 三種 typed decisions，並示範單張 GPU 微調、測試、AG2 Agent 與 Flask Web 展示。

## 1. 專案結構

```text
laya/
├── laya_zh_tw/
│   ├── 01_download_dataset.py
│   ├── 02_convert_dataset.py
│   ├── 03_train_laya.py
│   └── 04_test_laya.py
│
├── ag2_agent/
│   └── 01_coffee_agent_with_laya.py
│
├── flask_web/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── requirements.txt
└── README.md
```

## 2. 使用的資料集

Hugging Face：`renhehuang/coffee-order-zhtw`

- 臺灣繁體中文
- 約 2,900 筆多輪咖啡點餐對話
- Apache-2.0
- 包含基本點餐、資訊不足、修改、菜單外品項等情境
- 資料本身是 synthetic dialogue dataset

如果要使用完整資料，可以在 `01_download_dataset.py` 將：

```python
MAX_CONVERSATIONS = None
```

## 3. 資料怎麼轉成 Laya

原始資料大致是：

```text
system -> user -> assistant -> user -> assistant ...
```

`02_convert_dataset.py` 觀察原始 assistant 的下一句，使用簡單且可讀的規則產生 weak label：

- A：直接確認或整理訂單
- B：詢問缺少或不清楚的資訊
- C：目前無法提供，請顧客改選

接著，每個 `user -> assistant` turn 會產生三種 Laya typed decisions：

- `choice`：下一步選 A、B 或 C。
- `noul`：現在是否已經能直接確認訂單。
- `score`：目前訂單完成程度，從 0 到 2。

這些 action labels 不是原始資料集的人工作答，而是由原始回覆透過透明規則衍生的 weak labels。

## 4. 執行順序

在 `laya_zh_tw` 目錄依序執行：

```bash
python 01_download_dataset.py
python 02_convert_dataset.py
python 03_train_laya.py
python 04_test_laya.py
```

流程可以理解成：

```text
下載繁中資料
    ↓
轉成 typed decisions
    ↓
微調 Laya
    ↓
temperature calibration
    ↓
測試模型
```

`03_train_laya.py` 保留：

- RLCD proper-scoring reward
- Gaussian exploration
- GRPO-style group baseline
- soft cross-entropy guidance
- gradient checkpointing
- gradient accumulation
- held-out temperature calibration

`04_test_laya.py` 會計算：

- accuracy
- soft accuracy
- Brier score
- KL divergence
- total variation distance
- ECE
- score MAE
- within-one
- p50 latency

這個版本以單張 GPU 為主，因此沒有加入 multi-GPU DDP 包裝。

## 5. 學生最適合調整的參數

主要參數直接放在 `03_train_laya.py` 最上方，不使用額外設定檔。

### EPOCHS

模型把整份訓練資料看幾次。

- 調大：模型看資料更多次，但訓練比較久。
- 太大：可能開始記住訓練資料，而不是學到一般規律。

### MICRO_BATCH

一次放進 GPU 幾筆 decision。

- 調大：通常比較快，但需要更多 GPU memory。
- 如果出現 CUDA out of memory，可以先調小。

### GRAD_ACCUM

累積幾個小 batch 才真正更新一次模型。

- 調大：可以用較小顯存模擬較大的 effective batch。
- 代價：一次真正更新需要累積更多步。

例如：

```text
MICRO_BATCH = 2
GRAD_ACCUM = 32
```

effective batch 約為 64。

### GROUP_SIZE

RLCD 每次產生幾個探索版本來比較。

- 調大：模型一次比較更多候選。
- 代價：計算量也會增加。

### SIGMA_START / SIGMA_END

控制 Gaussian exploration 的幅度。

- 大一點：探索比較大膽。
- 小一點：比較接近目前模型的答案。
- 通常前期較大、後期較小。

### LR_ENCODER / LR_HEAD

learning rate，表示每次更新模型時改多少。

- 太大：可能學得不穩。
- 太小：可能學得很慢。

學生第一次實驗時，建議一次只修改一個參數，比較容易觀察影響。

## 6. 如果要改成英文或其他語言

這份版本只使用繁體中文資料，但架構本身可以抽換語言。

### 英文

主要修改三個位置：

1. `01_download_dataset.py`
   - 將 `DATASET_ID` 換成英文資料集。
2. `02_convert_dataset.py`
   - 將 `classify_reply()` 的繁中規則改成英文規則。
   - 將 instructions / criteria 改成英文。
3. `03_train_laya.py`
   - 將：

```python
BASE_MODEL = "convaiinnovations/laya-multilingual"
```

改成：

```python
BASE_MODEL = "convaiinnovations/laya"
```

### 其他語言

通常可以繼續使用：

```python
BASE_MODEL = "convaiinnovations/laya-multilingual"
```

再替換資料集、規則、instructions 與 criteria。

如果新的資料集本身已經有 class/action label，可以直接把原始 label 轉成 Laya target，不需要再使用 `classify_reply()`。

## 7. AG2 + Ollama Agent

Agent 範例位於：

```text
ag2_agent/01_coffee_agent_with_laya.py
```

執行：

```bash
cd ag2_agent
python 01_coffee_agent_with_laya.py
```

預設使用：

```text
qwen3.5:4b
```

如果 Ollama 不在本機，可以透過：

```bash
export OLLAMA_BASE_URL="你的 Ollama URL"
```

指定 Ollama 服務位置。

流程是：

```text
使用者輸入
    ↓
AG2 / Ollama
    ↓
呼叫 fine-tuned Laya
    ↓
choice / noul / score
    ↓
AG2 產生繁體中文回覆
```

## 8. Flask Web 展示

Web 範例位於：

```text
flask_web/
├── app.py
├── templates/
│   └── index.html
└── static/
    └── style.css
```

執行：

```bash
cd flask_web
python app.py
```

瀏覽器開啟：

```text
http://127.0.0.1:5000
```

Web 版直接呼叫 fine-tuned Laya，不經 AG2 或 Ollama，因此可以直接展示：

- `choice`：下一步應該確認、追問或請顧客改選。
- `noul`：目前是否已經可以直接確認訂單，以及 `P(true)`。
- `score`：目前訂單完成程度，以及各 level 的機率。
- 原始 Laya JSON 輸出。

模型路徑直接放在 `flask_web/app.py`：

```python
MODEL_DIR = PROJECT_ROOT / "laya_zh_tw" / "output" / "laya_coffee_zh_tw"
```

如果未來換成英文或其他語言，只要換掉 fine-tuned model、`SHOP_RULES` 與 `questions()` 裡的 instructions / criteria；Flask 架構本身不需要修改。
