"""
train_bert.py

使用 google-bert/bert-base-chinese 做繁體中文評論二元分類 Fine-tuning。

資料來源：

    data/bert/train.jsonl
    data/bert/valid.jsonl

整體流程：

    JSONL
        ↓
    Hugging Face Dataset
        ↓
    Tokenizer
        ↓
    Dynamic Padding
        ↓
    BERT
        ↓
    Sequence Classification Head
        ↓
    Cross-Entropy Loss
        ↓
    Backward
        ↓
    Gradient Accumulation
        ↓
    Optimizer Step
        ↓
    Validation
        ↓
    Best Checkpoint
        ↓
    output/bert_binary_clf/

這支程式保留原 notebook 的主要訓練設定，
並補上幾個較適合正式 fine-tuning 的處理：

1. Gradient accumulation 不代表「一定要把 learning rate 乘上 accumulation steps」。
   Trainer 會正確正規化 accumulated gradient。
   learning rate 應該被視為獨立的超參數。

2. Tokenization 不再先 padding=True 把資料預先補齊。
   改用 DataCollatorWithPadding 在每個 mini-batch 動態 padding，
   可以減少不必要的 padding token 與 GPU 計算。

3. 使用 validation loss 選擇最佳 checkpoint。

4. 加入 EarlyStoppingCallback。
   如果 eval_loss 連續多次沒有改善，就提前停止，
   避免模型在 training data 上繼續過度擬合。
"""

# =============================================================================
# 1. 匯入套件
# =============================================================================

from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers

from datasets import load_dataset

from sklearn.metrics import f1_score

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# =============================================================================
# 2. 模型與路徑設定
# =============================================================================

# Hugging Face 預訓練模型。
MODEL_NAME = "google-bert/bert-base-chinese"

# prepare_data.py 的輸出資料夾。
DATA_DIR = Path("data/bert")

TRAIN_FILE = "train.jsonl"
VALID_FILE = "valid.jsonl"

# Fine-tuned model 與 checkpoints 的輸出位置。
OUTPUT_DIR = Path("output/bert_binary_clf")


# =============================================================================
# 3. 模型設定
# =============================================================================

# 二元分類：
#
#     0 = negative
#     1 = positive
NUM_LABELS = 2

ID2LABEL = {
    0: "NEGATIVE",
    1: "POSITIVE",
}

LABEL2ID = {
    "NEGATIVE": 0,
    "POSITIVE": 1,
}

# BERT-base-chinese 的最大 context window 是 512 tokens。
MAX_SEQ_LENGTH = 512


# =============================================================================
# 4. 訓練設定
# =============================================================================

SEED = 42

NUM_TRAIN_EPOCHS = 3

# 一次 forward/backward 真正送進 GPU 的 examples 數。
PER_DEVICE_TRAIN_BATCH_SIZE = 32

# Validation 不需要 backward，
# 通常可以使用相同或更大的 batch。
PER_DEVICE_EVAL_BATCH_SIZE = 32

# Gradient accumulation。
#
# 目前：
#
#     train micro-batch = 32
#     accumulation      = 2
#     GPU 數量          = 1
#
# 所以 effective batch size 約為：
#
#     32 × 2 × 1 = 64
#
# 實際概念：
#
# 第 1 個 mini-batch：
#     32 筆
#     → forward
#     → backward
#     → 暫時累積 gradient
#
# 第 2 個 mini-batch：
#     32 筆
#     → forward
#     → backward
#     → 繼續累積 gradient
#
# 然後才：
#
#     optimizer.step()
#
# 因此大約每 64 筆 training examples 更新一次模型參數。
GRADIENT_ACCUMULATION_STEPS = 2

# 原 notebook 實際使用：
#
#     0.00005 * 2
#
# 數值就是：
#
#     1e-4
#
# 這裡保留同樣的數值，但要特別注意：
#
# gradient accumulation 並不會因為 Trainer 把 loss 做正規化，
# 就「必須」把 learning rate 乘上 accumulation steps。
#
# 1e-4 是一個獨立選擇的 learning-rate hyperparameter。
LEARNING_RATE = 1e-4

# 前 50 個 optimizer steps 做 linear warmup。
WARMUP_STEPS = 50

WEIGHT_DECAY = 0.01

LR_SCHEDULER_TYPE = "linear"

# 每隔多少 optimizer steps 做一次 validation。
#
# 目前 training data 約 6,212 筆，
# effective batch size 約為：
#
#     32 × 2 = 64
#
# 因此每個 epoch 大約：
#
#     6212 / 64
#     ≈ 97 optimizer steps
#
# 如果訓練 3 epochs，
# 總 optimizer steps 大約：
#
#     97 × 3
#     ≈ 291 steps
#
# 若 EVAL_STEPS = 70，
# 整個訓練大約只有 4 次 validation，
# 對 early stopping 而言觀察點太少。
#
# 因此這裡改成每 30 個 optimizer steps 做一次 validation，
# 整個 3 epochs 大約會有 9～10 次 validation。
EVAL_STEPS = 30

# checkpoint 儲存頻率必須和 evaluation strategy 相容。
#
# 這裡和 EVAL_STEPS 保持一致：
#
#     step 30
#       → validation
#       → checkpoint-30
#
#     step 60
#       → validation
#       → checkpoint-60
#
# 如此 load_best_model_at_end=True 才能穩定追蹤最佳 checkpoint。
SAVE_STEPS = 30

# 最多保留 2 個 checkpoints。
#
# 開啟 load_best_model_at_end=True 後，
# Trainer 會盡量保留最佳 checkpoint，
# 同時限制其他 checkpoints 不要無限制佔用磁碟。
SAVE_TOTAL_LIMIT = 2

# Early stopping patience。
#
# EARLY_STOPPING_PATIENCE = 3
#
# 表示如果連續 3 次 validation 的監控指標
# 都沒有比目前最佳值更好，
# 就提前停止 training。
#
# 本程式使用：
#
#     metric_for_best_model = "eval_loss"
#     greater_is_better = False
#
# 因此判斷標準是：
#
#     eval_loss 越低越好
#
# 例如：
#
#     step  30：eval_loss = 0.45
#     step  60：eval_loss = 0.38   ← 新 best
#     step  90：eval_loss = 0.40   ← 第 1 次沒改善
#     step 120：eval_loss = 0.42   ← 第 2 次沒改善
#     step 150：eval_loss = 0.41   ← 第 3 次沒改善
#
# 這時就會觸發 early stopping，
# 不一定需要把 NUM_TRAIN_EPOCHS=3 全部跑完。
EARLY_STOPPING_PATIENCE = 3

# 每 10 個 optimizer steps 印一次 training log。
LOGGING_STEPS = 10


# =============================================================================
# 5. Mixed Precision
# =============================================================================

# Tesla T4 適合 FP16。
#
# 如果不是 CUDA GPU，下面 main() 會自動關閉 fp16。
USE_FP16 = True

# T4 不使用 BF16。
USE_BF16 = False


# =============================================================================
# 6. 輔助函式
# =============================================================================

def print_gpu_info():
    """
    顯示目前 PyTorch 是否看到 GPU。
    """

    print("=" * 72)
    print(f"PyTorch：{torch.__version__}")
    print(f"Transformers：{transformers.__version__}")
    print(f"CUDA available：{torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")
        print(
            "CUDA capability："
            f"{torch.cuda.get_device_capability(0)}"
        )
        print(f"PyTorch CUDA：{torch.version.cuda}")

    print("=" * 72)


def compute_metrics(eval_pred):
    """
    Hugging Face Trainer 每次 evaluation 後呼叫。

    eval_pred.predictions：

        shape 通常是：
            [batch_size, num_labels]

        二元分類時例如：
            [
                [2.1, -0.5],
                [-1.0, 3.2],
                ...
            ]

        這些值是 logits，不是 probability。

    argmax(axis=-1)：

        對每一筆 example，
        取 logits 最大值所在的 class index。

    例如：

        [2.1, -0.5]
            → class 0

        [-1.0, 3.2]
            → class 1
    """

    labels = eval_pred.label_ids

    predictions = eval_pred.predictions

    # 某些模型/Trainer 設定可能回傳 tuple，
    # 這裡保留第一個 logits tensor。
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = np.argmax(
        predictions,
        axis=-1,
    )

    # binary F1：
    # 預設把 label=1 視為 positive class。
    f1 = f1_score(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    return {
        "f1": f1,
    }


# =============================================================================
# 7. 主流程
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Step 1：GPU 資訊
    # -------------------------------------------------------------------------

    print_gpu_info()

    # -------------------------------------------------------------------------
    # Step 2：確認資料
    # -------------------------------------------------------------------------

    train_path = DATA_DIR / TRAIN_FILE
    valid_path = DATA_DIR / VALID_FILE

    if not train_path.is_file():
        raise FileNotFoundError(
            f"找不到 training data：{train_path.resolve()}\n"
            "請先執行 python prepare_data.py"
        )

    if not valid_path.is_file():
        raise FileNotFoundError(
            f"找不到 validation data：{valid_path.resolve()}\n"
            "請先執行 python prepare_data.py"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Step 3：讀取 JSONL
    # -------------------------------------------------------------------------

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(valid_path),
        },
    )

    print(dataset)

    # -------------------------------------------------------------------------
    # Step 4：載入 tokenizer
    # -------------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # -------------------------------------------------------------------------
    # Step 5：Tokenization
    # -------------------------------------------------------------------------

    def tokenize_batch(batch):
        """
        只做 tokenization + truncation。

        注意這裡刻意不使用：

            padding=True

        因為我們會在真正組成 mini-batch 時，
        使用 DataCollatorWithPadding 做 dynamic padding。

        這樣每個 batch 只補到該 batch 最長序列，
        而不是預先補很多不必要的 PAD tokens。
        """

        return tokenizer(
            batch["sentences"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,

        # 原始文字不需要在 model.forward() 時傳進去，
        # tokenization 後可以移除 sentences 欄位。
        remove_columns=["sentences"],
    )

    # -------------------------------------------------------------------------
    # Step 6：Dynamic Padding
    # -------------------------------------------------------------------------

    # 假設同一個 mini-batch 三筆長度：
    #
    #     80 tokens
    #     130 tokens
    #     200 tokens
    #
    # DataCollatorWithPadding 只會補到：
    #
    #     200 tokens
    #
    # 而不是全部補到 MAX_SEQ_LENGTH=512。
    #
    # 對短文本資料通常可以省下不少計算。
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    # -------------------------------------------------------------------------
    # Step 7：建立 BERT sequence classifier
    # -------------------------------------------------------------------------

    # bert-base-chinese 原本只有 pretrained encoder。
    #
    # AutoModelForSequenceClassification 會在 BERT 上加一個
    # sequence-classification head。
    #
    # 因為 NUM_LABELS=2，
    # 最後 logits shape 會是：
    #
    #     [batch_size, 2]
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # -------------------------------------------------------------------------
    # Step 8：TrainingArguments
    # -------------------------------------------------------------------------

    # 只有真的有 CUDA 時才啟用 FP16。
    fp16_enabled = (
        USE_FP16
        and torch.cuda.is_available()
    )

    training_args = TrainingArguments(
        # Trainer 的 checkpoint 與最後輸出位置。
        #
        # 注意：
        # 舊版 Transformers 曾支援：
        #
        #     overwrite_output_dir=True
        #
        # 但目前 Colab 安裝的新版 TrainingArguments 已不接受
        # overwrite_output_dir 這個參數，因此這裡不再傳入。
        #
        # 如果你想完全重新訓練，最乾淨的方法是先自行刪除：
        #
        #     rm -rf output/bert_binary_clf
        #
        # 再執行本程式。
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=NUM_TRAIN_EPOCHS,

        per_device_train_batch_size=(
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            PER_DEVICE_EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,

        warmup_steps=WARMUP_STEPS,

        weight_decay=WEIGHT_DECAY,

        lr_scheduler_type=LR_SCHEDULER_TYPE,

        # 每固定 optimizer steps 做 validation。
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        # Evaluation 與 save 使用相同 strategy，
        # 讓 load_best_model_at_end 可以正常追蹤 checkpoint。
        save_strategy="steps",
        save_steps=SAVE_STEPS,

        save_total_limit=SAVE_TOTAL_LIMIT,

        logging_steps=LOGGING_STEPS,

        # 訓練結束後，自動把最佳 checkpoint 載回 trainer.model。
        #
        # 「訓練結束」可能有兩種情況：
        #
        # 1. 正常跑完 NUM_TRAIN_EPOCHS。
        # 2. EarlyStoppingCallback 提前停止。
        #
        # 無論是哪一種，
        # Trainer 最後都會把 validation 指標最佳的 checkpoint
        # 載回 trainer.model。
        load_best_model_at_end=True,

        # 用 validation loss 判斷最佳模型。
        #
        # EarlyStoppingCallback 也會依照這個指標判斷
        # 是否已經連續多次沒有改善。
        metric_for_best_model="eval_loss",

        # eval_loss 越低越好。
        #
        # 因此：
        #
        #     新 eval_loss < best eval_loss
        #
        # 才算 improvement。
        greater_is_better=False,

        fp16=fp16_enabled,
        bf16=USE_BF16,
        tf32=False,

        seed=SEED,
        data_seed=SEED,

        report_to="none",
    )

    # -------------------------------------------------------------------------
    # Step 9：建立 Trainer
    # -------------------------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=(
            tokenized_dataset["train"]
        ),

        eval_dataset=(
            tokenized_dataset["validation"]
        ),

        data_collator=data_collator,

        compute_metrics=compute_metrics,

        # -------------------------------------------------------------
        # Early Stopping
        # -------------------------------------------------------------
        #
        # EarlyStoppingCallback 會讀取：
        #
        #     metric_for_best_model
        #
        # 目前設定是：
        #
        #     eval_loss
        #
        # 所以每次 evaluation 後，
        # callback 會比較新的 eval_loss
        # 和目前最佳 eval_loss。
        #
        # 如果連續 EARLY_STOPPING_PATIENCE 次
        # 都沒有改善，
        # Trainer 就會提早結束。
        #
        # 注意：
        # Early stopping 不代表「最後一個 checkpoint」最好。
        #
        # 因為我們同時設定：
        #
        #     load_best_model_at_end=True
        #
        # 所以 training 結束後，
        # Trainer 仍會把最佳 checkpoint
        # 重新載回 trainer.model。
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=(
                    EARLY_STOPPING_PATIENCE
                )
            )
        ],
    )

    # -------------------------------------------------------------------------
    # Step 10：開始 fine-tuning
    # -------------------------------------------------------------------------

    trainer.train()

    # -------------------------------------------------------------------------
    # Step 11：顯示最佳 checkpoint
    # -------------------------------------------------------------------------

    print(
        "最佳 checkpoint："
        f"{trainer.state.best_model_checkpoint}"
    )

    print(
        "最佳 eval_loss："
        f"{trainer.state.best_metric}"
    )

    # -------------------------------------------------------------------------
    # Step 12：儲存最佳模型
    # -------------------------------------------------------------------------

    # 因為 load_best_model_at_end=True，
    # 此時 trainer.model 已經是最佳 checkpoint。
    trainer.save_model(
        str(OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(OUTPUT_DIR)
    )

    # trainer_state.json 包含：
    #
    # - log_history
    # - global_step
    # - best_metric
    # - best_model_checkpoint
    #
    # plot_bert_metrics.py 會使用它。
    trainer.save_state()

    print(
        "Final model directory："
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
