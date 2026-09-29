# Laya 繁體中文咖啡點餐教學版

## 1. 使用的資料集

Hugging Face：`renhehuang/coffee-order-zhtw`

- 臺灣繁體中文
- 2,939 筆多輪咖啡點餐對話
- Apache-2.0
- 包含基本點餐、資訊不足、修改、菜單外品項等情境
- 資料由 OllaForge 生成，因此是 synthetic dialogue dataset

要使用完整資料：

```python
MAX_CONVERSATIONS = None
```

## 2. 資料怎麼轉成 Laya

原資料是一段段：

```text
system -> user -> assistant -> user -> assistant ...
```

`02_convert_dataset.py` 觀察「原始 assistant 下一句」，用很容易看懂的規則產生 weak label：

- A：直接確認或整理訂單
- B：詢問缺少或不清楚的資訊
- C：目前無法提供，請顧客改選

接著每個 `user -> assistant` turn 產生三筆 Laya typed decisions：

- `choice`：A / B / C 下一步選哪個。
- `noul`：現在是否已經能直接確認訂單。
- `score`：目前可完成程度 0 / 1 / 2。

這些 action label 不是原始人工標註，而是從原始回覆透過透明規則衍生的 weak labels。

## 3. Colab 執行

先選 GPU runtime，例如 Tesla T4。

```bash
pip install "laya==0.3.20" datasets safetensors
```

依序執行：

```bash
cd laya_zh_tw
python 01_download_dataset.py
python 02_convert_dataset.py
python 03_train_laya.py
python 04_test_laya.py
```

`03_train_laya.py` 保留原方法的重要元素：

- RLCD proper-scoring reward
- Gaussian exploration
- GRPO-style group baseline
- soft cross-entropy guidance
- gradient checkpointing
- gradient accumulation
- held-out temperature calibration

`04_test_laya.py` 包含：

- accuracy
- soft accuracy
- Brier score
- KL divergence
- total variation distance
- ECE
- score MAE
- within-one
- p50 latency

這是單張 T4 教學版，所以沒有加入多 GPU DDP 包裝。DDP 是多卡執行基礎設施，不會改變上述 RLCD / calibration / benchmark 方法；若需要 2 張以上 GPU，可回到官方 notebook 的 DDP wrapper。

## 4. 學生最適合改的參數

直接打開 `03_train_laya.py` 最上方。

### EPOCHS

模型把整份訓練資料看幾次。

- 大一點：可能學得更多，也更慢。
- 太大：可能只是在背訓練資料。

### MICRO_BATCH

一次放進 GPU 幾筆 decision。

- 大一點：通常比較快，但吃更多 GPU memory。
- 如果出現 CUDA out of memory，先把它調小。

### GRAD_ACCUM

累積幾個小 batch 才更新一次模型。

- 大一點：可以用小顯存模擬較大的 effective batch。
- 代價：一次真正更新要等比較久。

本例：

```text
MICRO_BATCH = 2
GRAD_ACCUM = 32
effective batch 約為 64
```

### GROUP_SIZE

RLCD 每次製造幾個探索版本來比較。

- 大一點：比較的候選更多。
- 但計算量也會增加。

### SIGMA_START / SIGMA_END

控制探索時「搖動答案」的幅度。

- 大：嘗試得比較大膽。
- 小：比較接近目前模型的答案。
- 通常前期大、後期小。

### LR_ENCODER / LR_HEAD

learning rate，代表每次更新模型要改多少。

- 太大：容易學得不穩。
- 太小：可能學得很慢。

學生第一次實驗建議每次只改一個參數。

## 5. 如果要改成英文或其他語言

不需要重新設計整套程式，主要替換幾個位置。

### 英文

1. `01_download_dataset.py`
   - 把 `DATASET_ID` 換成英文資料集。
2. `02_convert_dataset.py`
   - 把 `classify_reply()` 的繁中關鍵字規則換成英文規則。
   - 把 instructions / criteria 改成英文。
3. `03_train_laya.py`
   - 把：

```python
BASE_MODEL = "convaiinnovations/laya-multilingual"
```

改成：

```python
BASE_MODEL = "convaiinnovations/laya"
```

### 其他語言

保留：

```python
BASE_MODEL = "convaiinnovations/laya-multilingual"
```

再替換資料集、規則和 prompts 即可。

如果新資料集本身就有 class/action label，甚至可以拿掉 `classify_reply()`，直接把原始 label 轉成 Laya target，會更乾淨。

## 6. AG2 + Ollama

AG2 範例放在獨立資料夾：

```bash
pip install "ag2[ollama]"
cd ag2_agent
python 01_coffee_agent_with_laya.py
```

預設使用：

```text
qwen3.5:4b
```

如果 Ollama 在別台機器或 Colab + ngrok：

```bash
export OLLAMA_BASE_URL="你的 Ollama URL"
```

Agent 收到咖啡訂單後會先呼叫 fine-tuned Laya tool，取得 `choice / noul / score`，再由 Ollama 產生簡短的繁體中文自然語言回覆。

## 7. Flask Web 展示

除了命令列測試與 AG2 Agent，另外提供一個很簡單的 Web 展示：

```text
flask_web/
├── app.py
├── templates/
│   └── index.html
└── static/
    └── style.css
```

這個 Web 版**直接呼叫 fine-tuned Laya**，不經過 AG2 或 Ollama，因此可以直接看到：

- `choice`：下一步應該確認、追問或請顧客改選。
- `noul`：目前是否已經可以直接確認訂單，顯示 P(true)。
- `score`：目前訂單可完成程度，以及各 score level 的機率。

先安裝 Flask：

```bash
pip install flask
```

如果前面還沒有安裝 Laya：

```bash
pip install "laya==0.3.20"
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

如果是在 Colab，Flask 會監聽 `0.0.0.0:5000`；要從外部瀏覽器連入，可以再使用你自己的 ngrok 或其他 tunnel 將 5000 port 暴露出去。

模型路徑直接寫在 `flask_web/app.py` 最上方：

```python
MODEL_DIR = PROJECT_ROOT / "laya_zh_tw" / "output" / "laya_coffee_zh_tw"
```

如果未來換成英文或其他語言，只要同時換掉 fine-tuned model、`SHOP_RULES` 與 `questions()` 裡的 instructions / criteria；Flask 架構本身不需要修改。

> `python app.py` 使用的是 Flask development server，適合課堂與展示，不建議直接當正式公開服務使用。
