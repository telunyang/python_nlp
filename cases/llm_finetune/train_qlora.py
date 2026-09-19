"""
使用 4-bit QLoRA 微調繁體中文多輪對話模型。

這個程式設計給「第一次接觸 PyTorch、Transformer、LoRA 或 QLoRA」的人閱讀。
因此除了執行訓練之外，也盡量在程式碼旁邊說明：

1. 每一段程式在做什麼。
2. 為什麼需要這樣做。
3. 它和 GPU 記憶體、模型參數、梯度更新之間的關係。
4. 常見設定值代表什麼。
5. 在 Tesla T4 上為什麼要使用 FP16，而不是 BF16。

整體流程可以先理解成：

    JSONL 對話資料
        ↓
    Tokenizer 把文字轉成 token IDs
        ↓
    載入 Qwen2.5-0.5B-Instruct
        ↓
    Base model 權重以 4-bit NF4 儲存
        ↓
    prepare_model_for_kbit_training()
        ↓
    加上 LoRA adapter
        ↓
    只有少量 LoRA 參數參與訓練
        ↓
    SFTTrainer 執行 forward / loss / backward / optimizer step
        ↓
    定期 evaluation
        ↓
    儲存 checkpoint
        ↓
    early stopping
        ↓
    載入 validation loss 最低的最佳 checkpoint
        ↓
    儲存最後的 LoRA adapter 與 tokenizer

這裡的「QLoRA」可以先簡單理解成：

    Q = Quantization
        Base model 使用低精度 4-bit 表示，降低 GPU 記憶體需求。

    LoRA = Low-Rank Adaptation
        不直接修改完整 base model，而是在部分線性層旁邊加入少量可訓練參數。

例如原始模型可能有約 5 億個參數，但真正需要更新的 LoRA 參數可能只有幾百萬個。
這可以大幅降低微調所需的 GPU 記憶體。

本程式特別針對 Google Colab Tesla T4 設定：

    4-bit base model + FP16 compute + FP32 LoRA parameters

原因是 Tesla T4 不適合直接使用 BF16 mixed precision。
如果 trainable parameters 混入 BF16，PyTorch AMP GradScaler 可能出現：

    NotImplementedError:
    "_amp_foreach_non_finite_check_and_unscale_cuda"
    not implemented for 'BFloat16'

因此本程式會在訓練開始前主動檢查 trainable parameters 的 dtype。
"""

from collections import Counter
from pathlib import Path

# torch 是 PyTorch 的核心套件。
#
# 神經網路訓練時，它主要負責：
# - Tensor 計算
# - GPU / CUDA 運算
# - 自動微分 autograd
# - gradient backward
#
# 在本程式中，我們主要用它：
# 1. 檢查是否有 NVIDIA GPU。
# 2. 取得 GPU 型號與 CUDA capability。
# 3. 指定 FP16 / FP32 / BF16 等資料型態。
import torch

# Hugging Face Datasets。
#
# load_dataset() 可以把 JSON / JSONL 等資料讀成 Dataset。
#
# 例如 train.jsonl：
#
# {"prompt": [...], "completion": [...]}
# {"prompt": [...], "completion": [...]}
#
# 每一行都是一筆 training example。
from datasets import load_dataset

# PEFT = Parameter-Efficient Fine-Tuning。
#
# LoraConfig：
#   定義 LoRA adapter 的 rank、alpha、dropout 等設定。
#
# get_peft_model：
#   真正把 LoRA adapter 掛到 base model 上。
#
# prepare_model_for_kbit_training：
#   在 4-bit / 8-bit quantized model 上進行 PEFT 訓練前的重要準備步驟。
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# AutoModelForCausalLM：
#   載入 causal language model。
#   Qwen2.5-Instruct 就屬於這類模型。
#
# AutoTokenizer：
#   將文字轉換成 token IDs，也負責把 token IDs 解碼回文字。
#
# BitsAndBytesConfig：
#   設定 4-bit / 8-bit quantization。
#
# EarlyStoppingCallback：
#   validation 指標長時間不再改善時，自動停止訓練。
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)

# TRL = Transformer Reinforcement Learning library。
#
# 雖然名稱中有 RL，但 SFTTrainer 是用來做 Supervised Fine-Tuning。
#
# SFTConfig：
#   設定訓練參數。
#
# SFTTrainer：
#   幫我們處理：
#   - tokenization
#   - batching
#   - forward
#   - loss
#   - backward
#   - optimizer
#   - evaluation
#   - checkpoint
from trl import SFTConfig, SFTTrainer


# =============================================================================
# 1. 模型與路徑設定
# =============================================================================


# Hugging Face Hub 上的模型名稱。
#
# Qwen/Qwen2.5-0.5B-Instruct：
# - Qwen2.5 系列
# - 約 0.5B，也就是約 5 億參數
# - Instruct 版本已經做過 instruction/chat tuning
#
# 為什麼選 Instruct，而不是 Base？
#
# Base model：
#   主要學習「下一個 token 預測」。
#
# Instruct model：
#   已經學過 user / assistant 的對話格式，
#   比較適合直接拿來做聊天資料的 SFT。
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


# 訓練資料所在目錄。
#
# 這個路徑是相對於 train_qlora.py 的位置。
#
# 例如檔案結構：
#
# llm_finetune/
# ├── train_qlora.py
# └── data/
#     └── sft/
#         ├── train.jsonl
#         └── valid.jsonl
#
# 那麼 DATA_DIR 設成 "data/sft" 即可。
DATA_DIR = "data/sft"


# Training split 的檔名。
#
# 訓練資料會用來真正計算 gradient，更新 LoRA parameters。
TRAIN_FILE = "train.jsonl"


# Validation split 的檔名。
#
# Validation 資料「不拿來更新參數」。
# 它主要用於：
# - 檢查模型是否過度擬合
# - 計算 eval_loss
# - 找出最佳 checkpoint
# - early stopping
VALID_FILE = "valid.jsonl"


# 模型輸出目錄。
#
# 訓練過程中可能會產生：
#
# outputs/qwen25_05b_lora/
# ├── checkpoint-50/
# ├── checkpoint-100/
# ├── checkpoint-150/
# ├── adapter_config.json
# ├── adapter_model.safetensors
# └── tokenizer files...
#
# 注意：
# 如果在 Google Colab 使用 /content，
# Runtime 被刪除後，檔案通常也會一起消失。
# 正式實驗請考慮 Google Drive 或其他持久化儲存。
OUTPUT_DIR = "outputs/qwen25_05b_lora"


# 控制是否從 checkpoint 繼續訓練。
#
# 三種常見用法：
#
# 1. 從頭開始：
#
# RESUME_FROM_CHECKPOINT = None
#
# 2. 自動尋找 OUTPUT_DIR 中最新 checkpoint：
#
# RESUME_FROM_CHECKPOINT = True
#
# 3. 指定 checkpoint：
#
# RESUME_FROM_CHECKPOINT = "outputs/qwen25_05b_lora/checkpoint-200"
#
# checkpoint 不只包含 LoRA 權重，
# Trainer 還可以恢復 optimizer state、scheduler state、
# global step，以及 callback state。
RESUME_FROM_CHECKPOINT = None


# 某些 Hugging Face 模型 repository 會提供自訂 Python 程式碼。
#
# trust_remote_code=True：
#   允許執行 repository 中的自訂程式碼。
#
# Qwen2.5 一般不需要這麼做，所以保持 False 比較安全。
TRUST_REMOTE_CODE = False


# =============================================================================
# 2. 訓練設定
# =============================================================================


# 每筆樣本最多允許多少 tokens。
#
# Token 不是「字」也不是「詞」的完全等價物。
#
# 例如：
#
# "Hello world"
#
# 可能被 tokenizer 切成數個 token。
#
# MAX_LENGTH=768 表示：
# 每筆完整訓練序列最多保留 768 tokens。
#
# 如果原始資料超過這個長度，後面的內容可能被截斷。
#
# 長度越大：
# - 可以保留更多上下文
# - GPU 記憶體用量也會增加
# - attention 計算量也會增加
#
# 對 T4 16 GB 等級的 GPU，
# 768 是相對保守的教學設定。
MAX_LENGTH = 768


# 最多訓練幾個 epoch。
#
# 一個 epoch 的意思是：
# 模型大致看完整個 training dataset 一次。
#
# 例如：
# training dataset = 4,668 筆
# EPOCHS = 5
#
# 表示模型最多會重複看這批資料約 5 次。
#
# 但因為有 early stopping，
# 實際上可能在第 3 或第 4 個 epoch 就停止。
EPOCHS = 5


# Learning rate，學習率。
#
# 可以把它想成每次 optimizer 更新參數時的「步伐大小」。
#
# 太大：
# - loss 可能震盪
# - 訓練不穩定
#
# 太小：
# - 模型學得很慢
# - 在有限 epoch 內可能學不到足夠變化
#
# 1e-4 = 0.0001
#
# 對 LoRA SFT 是常見的起始值之一。
LEARNING_RATE = 1e-4


# 每張 GPU 一次真正送進模型多少筆 training examples。
#
# 這裡進一步提高到 8，目的是讓 T4 一次平行處理更多樣本，
# 提高 GPU 利用率，並進一步使用目前仍大量閒置的顯存。
#
# batch size 越大：
# - GPU 平行度通常越好
# - 顯存使用量通常也會提高
#
# 目前設定為 8。
# 如果實際訓練仍只有很低的 VRAM 使用量，且沒有 OOM，
# 下一步才考慮 batch=16。
PER_DEVICE_TRAIN_BATCH_SIZE = 8


# Validation 時一次處理多少筆資料。
#
# Evaluation 不需要 backward，也不需要儲存訓練用 gradient，
# 因此通常可以使用比 training 更大的 batch。
#
# 這裡設成 16，可進一步減少 validation 的 batch 次數。
#
# 例如 validation 有 520 筆資料：
# - eval batch = 1   → 約 520 個 batches
# - eval batch = 8   → 約 65 個 batches
# - eval batch = 16  → 約 33 個 batches
#
# Evaluation 不需要 backward，因此通常比 training 更能承受較大 batch。
PER_DEVICE_EVAL_BATCH_SIZE = 16


# Gradient Accumulation。
#
# 目前設定：
#
#     PER_DEVICE_TRAIN_BATCH_SIZE = 8
#     GRADIENT_ACCUMULATION_STEPS = 16
#
# GPU 每次處理 8 筆 training examples，
# 執行一次 forward 與 backward，
# 並暫時累積 gradient。
#
# 這個過程會連續執行 16 個 mini-batches，
# 累積完成後才執行一次 optimizer.step()
# 更新 LoRA parameters。
#
# 因此在單張 GPU 的情況下，
# effective batch size 約為：
#
#     PER_DEVICE_TRAIN_BATCH_SIZE
#     × GRADIENT_ACCUMULATION_STEPS
#
#     = 8 × 16
#     = 128
#
# 可以理解成：
#
#     8 筆 × 16 個 mini-batches
#     = 約 128 筆 training examples
#
# 才進行一次 optimizer update。
#
# 注意：
# GRADIENT_ACCUMULATION_STEPS = 16
# 表示累積 16 個 mini-batches，
# 不是累積 16 筆資料。
#
# 此設定相較 effective batch size = 16，
# 每次參數更新使用更多樣本，因此 gradient 通常較平滑，
# 但每個 epoch 的 optimizer update 次數也會明顯減少。
GRADIENT_ACCUMULATION_STEPS = 16


# Gradient Checkpointing。
#
# 一般 forward 時，神經網路會保存 intermediate activations，
# backward 時計算 gradient 會直接使用這些值。
#
# Gradient checkpointing=True 時：
# - 少保存部分 activation
# - backward 時重新計算
# - 可以省顯存
# - 但會讓訓練變慢
#
# 目前實測只有約 4～5 GB / 15 GB VRAM 被使用，
# 因此這版先關閉 gradient checkpointing，
# 用較高顯存換取較少 recomputation 與更快速度。
#
# 如果執行後發生 CUDA out of memory，可先改回 True。
GRADIENT_CHECKPOINTING = False


# Optimizer。
#
# AdamW 是 Transformer 很常見的 optimizer。
#
# paged_adamw_8bit：
# - AdamW 的 optimizer state 使用更省記憶體的表示
# - paged 機制有助於降低記憶體尖峰
#
# 這是 bitsandbytes 常見的 QLoRA 搭配方式。
OPTIM = "paged_adamw_8bit"


# 每隔多少 optimizer steps 顯示一次 training log。
#
# 例如：
#
# LOGGING_STEPS = 10
#
# 可能看到：
#
# {'loss': 1.82, 'learning_rate': ..., 'epoch': ...}
#
# 注意：
# 這裡的 step 是 optimizer update step，
# 不一定等同於每一筆 mini-batch。
LOGGING_STEPS = 10


# Evaluation 與 checkpoint 的觸發策略。
#
# "steps"：
#   每隔固定 optimizer steps 評估 / 存檔。
#
# "epoch"：
#   每個 epoch 結束才評估 / 存檔。
#
# 對 Colab 這種可能中斷的環境，
# "steps" 通常更容易保留訓練進度。
CHECK_STRATEGY = "steps"


# 每隔多少 optimizer steps：
# - 做一次 validation
# - 存一次 checkpoint
#
# 例如 CHECK_STEPS = 50：
#
# step 50
#   → eval
#   → checkpoint-50
#
# step 100
#   → eval
#   → checkpoint-100
CHECK_STEPS = 50


# 最多保留幾份 checkpoint。
#
# 如果設成 3，
# Trainer 會盡量避免 checkpoint 無限制累積。
#
# 最佳 checkpoint 會被優先保留。
CHECKPOINT_LIMIT = 3


# Early stopping 的耐心值。
#
# EARLY_STOPPING_PATIENCE = 3
#
# 表示：
# 如果連續 3 次 evaluation，
# eval_loss 都沒有變得更好，
# 就停止訓練。
#
# 例如：
#
# eval 1：1.20
# eval 2：1.10   ← 改善
# eval 3：1.12   ← 沒改善
# eval 4：1.15   ← 沒改善
# eval 5：1.14   ← 沒改善
#
# 此時累積 3 次沒有改善，
# 就可能觸發 early stopping。
EARLY_STOPPING_PATIENCE = 3


# completion_only_loss=True：
#
# 對 prompt-completion / chat 類型資料，
# 只針對 assistant 要回答的部分計算 loss。
#
# 舉例：
#
# User:
#   請解釋 BM25。
#
# Assistant:
#   BM25 是一種排序函數...
#
# 我們希望模型主要學習「Assistant 應該怎麼回答」，
# 而不是要求它重新預測 User prompt。
COMPLETION_ONLY_LOSS = True


# Qwen chat template 使用的對話結束 token。
#
# <|im_end|>
#
# 可以把它理解成：
# 「這一個 chat message 到這裡結束」。
#
# 正確的 EOS token 對模型學習何時停止回答很重要。
EOS_TOKEN = "<|im_end|>"


# Hugging Face Trainer 可以把訓練資訊傳到：
# - Weights & Biases
# - TensorBoard
# - MLflow
# 等服務。
#
# "none" 表示這個教學程式不連接外部 tracking service。
REPORT_TO = "none"


# Random seed。
#
# 訓練中有很多隨機行為，例如：
# - dataset shuffle
# - dropout
# - 某些初始化
#
# 固定 seed 可以提高可重現性。
#
# 但要注意：
# 深度學習在不同 GPU、CUDA、library version 下，
# 即使 seed 相同，也不保證每一個 floating-point 結果完全一致。
SEED = 42


# =============================================================================
# 目前這版的 T4 效能設定摘要
# =============================================================================
#
# Train micro-batch        = 8
# Gradient accumulation    = 2
# Effective train batch    = 8 × 2 = 16
# Eval batch               = 16
# Gradient checkpointing   = False
#
# 這樣保留原本 effective batch size = 16，
# 但讓 GPU 一次處理更多資料，並關掉 activation recomputation。
#
# 執行時建議另外開一個 terminal：
#
#     watch -n 1 nvidia-smi
#
# 觀察：
# - Memory-Usage
# - GPU-Util
#
# 若 VRAM 約 12～13.5 GB 且 GPU-Util 常接近 90～100%，通常相當理想。
# 若發生 CUDA OOM，建議依序退階：
#
# 1. 先把：
#
#     GRADIENT_CHECKPOINTING = True
#
# 2. 若仍 OOM，再改回：
#
#     PER_DEVICE_TRAIN_BATCH_SIZE = 4
#     GRADIENT_ACCUMULATION_STEPS = 4
#
# 3. 若還是 OOM，再使用：
#
#     PER_DEVICE_TRAIN_BATCH_SIZE = 2
#     GRADIENT_ACCUMULATION_STEPS = 8
#
# 以上三種配置的 effective batch size 都仍然是 16。
#
# =============================================================================
# 3. LoRA 設定
# =============================================================================


# LoRA rank，通常寫成 r。
#
# LoRA 不直接學習完整的大矩陣更新 ΔW，
# 而是把它近似成兩個較小矩陣：
#
# ΔW ≈ B × A
#
# rank r 決定中間維度大小。
#
# r 越大：
# - trainable parameters 越多
# - 表達能力通常越高
# - 記憶體與計算需求也增加
#
# r=8 是教學與小模型常見的起點。
LORA_R = 8


# LoRA alpha。
#
# LoRA 更新通常會經過 scaling：
#
# scaling = alpha / r
#
# 現在：
#
# alpha = 16
# r = 8
#
# 所以：
#
# scaling = 16 / 8 = 2
#
# 可以把它理解成 LoRA 更新強度的縮放係數之一。
LORA_ALPHA = 16


# LoRA dropout。
#
# 0.05 = 5%
#
# 在訓練時隨機丟棄一小部分 LoRA 路徑訊號，
# 有助於降低過度擬合。
LORA_DROPOUT = 0.05


# LoRA 要插入哪些 module。
#
# "all-linear"：
# 對模型中的各種 linear layers 套用 LoRA。
#
# 對 Transformer 而言，
# attention 與 feed-forward network 裡都有很多 linear layers。
#
# 好處：
# 不需要手動寫死：
#
# q_proj
# k_proj
# v_proj
# o_proj
# gate_proj
# up_proj
# down_proj
#
# 這對教學程式較簡潔。
LORA_TARGET_MODULES = "all-linear"


# =============================================================================
# 4. 輔助函式
# =============================================================================


def resolve_path(path_value: str) -> Path:
    """
    將相對路徑轉成「相對於 train_qlora.py」的實際路徑。

    為什麼不直接使用 Path("data/sft")？

    因為使用者可能從不同工作目錄執行：

        cd /content
        python llm_finetune/train_qlora.py

    如果所有相對路徑都依賴目前 shell 的 cwd，
    很容易找不到資料。

    這個函式固定以 train_qlora.py 所在位置為基準。
    """

    # __file__ 是目前 Python 檔案本身的路徑。
    #
    # resolve()：
    # 轉成 absolute path。
    #
    # parent：
    # 取得 train_qlora.py 所在資料夾。
    script_dir = Path(__file__).resolve().parent

    # 將字串轉成 Path object。
    path = Path(path_value)

    # 如果使用者本來就給 absolute path，
    # 例如：
    #
    # /content/llm_finetune/data/sft
    #
    # 就直接使用，不需要再拼接。
    if path.is_absolute():
        return path

    # 如果是相對路徑，
    # 就以 train_qlora.py 的資料夾為起點。
    return script_dir / path


def print_gpu_info() -> None:
    """
    檢查 CUDA GPU，並列印目前硬體與 precision 資訊。

    這一步非常重要，因為本程式是針對 NVIDIA Tesla T4 設計。
    """

    # torch.cuda.is_available() 會確認 PyTorch 是否真的能看到 CUDA GPU。
    #
    # 如果回傳 False，
    # 即使主機上看似有 GPU，
    # Python 目前也可能沒有正確 CUDA runtime。
    if not torch.cuda.is_available():
        raise RuntimeError(
            "找不到 CUDA GPU，請先啟用 NVIDIA GPU。"
        )

    # 取得第 0 張 GPU 的名稱。
    #
    # 在 Colab T4 上通常會得到：
    #
    # Tesla T4
    gpu_name = torch.cuda.get_device_name(0)

    # Compute capability 是 NVIDIA GPU 架構能力的版本。
    #
    # T4 是：
    #
    # (7, 5)
    #
    # Ampere，例如 A100，通常是 8.x。
    capability = torch.cuda.get_device_capability(0)

    print("=" * 80)
    print(f"GPU：{gpu_name}")
    print(f"CUDA capability：{capability}")

    # torch.version.cuda 顯示目前 PyTorch build 使用的 CUDA 版本。
    print(f"PyTorch CUDA：{torch.version.cuda}")

    # 這裡顯示 PyTorch 判斷的 BF16 支援狀態。
    #
    # 不論輸出結果如何，
    # 本程式針對 T4 仍固定使用 FP16，
    # 以避免 AMP + BF16 的相容性問題。
    # PyTorch 新版的 is_bf16_supported() 預設：
    #
    #     including_emulation=True
    #
    # 因此像 Tesla T4 這種沒有「原生 BF16 Tensor Core」
    # 的 GPU，也可能因 CUDA / PyTorch 可以用軟體方式處理
    # BF16 tensor，而回傳 True。
    #
    # 所以我們同時列出：
    #
    # 1. BF16 software/emulation support
    # 2. BF16 native hardware support
    #
    # 對本程式來說，真正重要的是 native hardware support。
    bf16_supported = torch.cuda.is_bf16_supported()

    try:
        bf16_native = torch.cuda.is_bf16_supported(
            including_emulation=False
        )
    except TypeError:
        # 舊版 PyTorch 沒有 including_emulation 參數時，
        # 就用 compute capability 判斷。
        #
        # NVIDIA Ampere 與更新架構通常是 major >= 8。
        bf16_native = capability[0] >= 8

    print(
        "BF16 supported "
        f"(包含軟體模擬)：{bf16_supported}"
    )

    print(
        "BF16 native hardware support："
        f"{bf16_native}"
    )

    print(
        "本訓練腳本固定使用 FP16 compute，"
        "以相容 Tesla T4。"
    )

    print("=" * 80)


def cast_trainable_parameters_to_fp32(
    model: torch.nn.Module,
) -> None:
    """
    將所有 requires_grad=True 的參數轉成 FP32。

    這裡只會影響「真正要訓練的參數」。

    QLoRA 中：
    - Base model 大部分權重是 4-bit，而且 frozen。
    - LoRA adapter 是少量 trainable parameters。

    因此這個操作不會把整個 4-bit 模型反量化成 FP32。

    為什麼讓 LoRA parameters 使用 FP32？

    因為：
    1. Trainable parameters 數量很少。
    2. FP32 gradient 更新較穩定。
    3. 可以避免 T4 上 BF16 gradient 和 AMP GradScaler 的相容性問題。

    例如：

        LoRA trainable params ≈ 4.4M

    使用 FP32：

        4.4M × 4 bytes
        ≈ 17.6 MB

    相對於 16 GB 等級 GPU，
    額外成本非常小。
    """

    # model.parameters() 會逐一取得模型中的 Parameter。
    for param in model.parameters():

        # requires_grad=True 表示：
        # backward 時 PyTorch 需要替它計算 gradient。
        #
        # frozen base model 通常是 False。
        if (
            param.requires_grad
            and param.dtype != torch.float32
        ):
            # 只轉 trainable parameter 的 data。
            param.data = param.data.to(
                torch.float32
            )


def print_trainable_dtypes(
    model: torch.nn.Module,
) -> None:
    """
    顯示 trainable parameters 使用哪些 dtype。

    本程式理想狀況應該看到：

        Trainable parameter dtypes:
          torch.float32: ...

    如果看到 torch.bfloat16，
    就代表還有 BF16 trainable parameters 混進來，
    本程式會直接停止。
    """

    # Counter 用來統計：
    # 每一種 dtype 出現幾個 Parameter tensor。
    #
    # 注意：
    # 這裡統計的是 parameter tensors 的數量，
    # 不是 scalar parameter 總數。
    dtypes = Counter(
        str(param.dtype)
        for param in model.parameters()
        if param.requires_grad
    )

    print("Trainable parameter dtypes:")

    for dtype, count in sorted(dtypes.items()):
        print(f"  {dtype}: {count}")

    # 找出所有：
    #
    # requires_grad=True
    # 且 dtype=torch.bfloat16
    #
    # 的參數名稱。
    bad_params = [
        name
        for name, param in model.named_parameters()
        if (
            param.requires_grad
            and param.dtype == torch.bfloat16
        )
    ]

    # 只要找到任何 BF16 trainable parameter，
    # 就在真正開始訓練前停止。
    #
    # 這比等到 optimizer step 才 crash 更容易除錯。
    if bad_params:

        # 最多列出前 20 個 problematic parameter names，
        # 避免 terminal 被大量輸出塞滿。
        preview = "\n".join(
            f"  - {name}"
            for name in bad_params[:20]
        )

        raise RuntimeError(
            "偵測到 BF16 trainable parameters；"
            "Tesla T4 不應使用此設定。\n"
            f"{preview}"
        )


# =============================================================================
# 5. 主訓練流程
# =============================================================================


def main() -> None:
    """
    執行完整 QLoRA Supervised Fine-Tuning。

    流程：

    1. 檢查 GPU
    2. 檢查資料
    3. 載入 dataset
    4. 載入 tokenizer
    5. 建立 4-bit quantization config
    6. 載入 base model
    7. 準備 k-bit training
    8. 建立 LoRA adapter
    9. 建立 SFTConfig
    10. 建立 SFTTrainer
    11. train
    12. 儲存最佳模型與 tokenizer
    """

    # -------------------------------------------------------------------------
    # Step 1：確認 GPU
    # -------------------------------------------------------------------------

    print_gpu_info()


    # -------------------------------------------------------------------------
    # Step 2：解析資料與輸出路徑
    # -------------------------------------------------------------------------

    data_dir = resolve_path(DATA_DIR)

    # data/sft/train.jsonl
    train_file = data_dir / TRAIN_FILE

    # data/sft/valid.jsonl
    valid_file = data_dir / VALID_FILE

    output_dir = resolve_path(OUTPUT_DIR)

    # parents=True：
    # 如果上層資料夾不存在，也一起建立。
    #
    # exist_ok=True：
    # 如果資料夾已經存在，不要報錯。
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -------------------------------------------------------------------------
    # Step 3：確認 training / validation files 存在
    # -------------------------------------------------------------------------

    if not train_file.is_file():
        raise FileNotFoundError(
            f"找不到 train dataset：{train_file}"
        )

    if not valid_file.is_file():
        raise FileNotFoundError(
            f"找不到 validation dataset：{valid_file}"
        )


    # -------------------------------------------------------------------------
    # Step 4：處理 checkpoint resume
    # -------------------------------------------------------------------------

    resume_from_checkpoint = (
        RESUME_FROM_CHECKPOINT
    )

    # 如果使用字串指定 checkpoint，
    # 將它解析成相對於 train_qlora.py 的完整路徑。
    if (
        isinstance(
            resume_from_checkpoint,
            str,
        )
        and resume_from_checkpoint
    ):
        resume_from_checkpoint = str(
            resolve_path(
                resume_from_checkpoint
            )
        )


    # -------------------------------------------------------------------------
    # Step 5：載入 dataset
    # -------------------------------------------------------------------------

    # Hugging Face load_dataset("json") 可以讀 JSONL。
    #
    # 回傳結果概念上像：
    #
    # DatasetDict({
    #     train: Dataset(...)
    #     validation: Dataset(...)
    # })
    data = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(valid_file),
        },
    )


    # -------------------------------------------------------------------------
    # Step 6：載入 tokenizer
    # -------------------------------------------------------------------------

    # Tokenizer 負責：
    #
    # 文字：
    # "你好"
    #
    # ↓ tokenize
    #
    # token IDs：
    # [xxxxx, xxxxx, ...]
    #
    # 模型實際處理的是 token IDs，
    # 不是 Python 字串。
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    # Padding：
    #
    # 同一個 batch 中，
    # 每筆序列長度可能不同。
    #
    # 例如：
    #
    # A: [10, 20, 30]
    # B: [40, 50]
    #
    # padding 後可能變成：
    #
    # A: [10, 20, 30]
    # B: [40, 50, PAD]
    #
    # 如果 tokenizer 沒有專門 pad token，
    # 就沿用 eos token。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )


    # -------------------------------------------------------------------------
    # Step 7：建立 4-bit QLoRA quantization 設定
    # -------------------------------------------------------------------------

    quant_config = BitsAndBytesConfig(

        # Base model 權重以 4-bit 載入。
        #
        # 一般 FP16：
        # 每個參數約 2 bytes。
        #
        # 4-bit：
        # 理論上每個參數約 0.5 bytes，
        # 另外還有 quantization metadata。
        #
        # 這是 QLoRA 節省記憶體的主要來源。
        load_in_4bit=True,

        # NF4 = NormalFloat4。
        #
        # 這是 bitsandbytes / QLoRA 常用的
        # 4-bit quantization format，
        # 適合神經網路權重分布。
        bnb_4bit_quant_type="nf4",

        # Double quantization：
        #
        # 不只量化模型權重，
        # 也進一步量化部分 quantization constants，
        # 以再降低一些記憶體。
        bnb_4bit_use_double_quant=True,

        # 非常重要：
        #
        # 權重「儲存」是 4-bit，
        # 但矩陣乘法實際計算仍需要較高精度。
        #
        # 這裡指定 compute dtype = FP16。
        #
        # Tesla T4 使用 FP16，
        # 不使用 BF16。
        bnb_4bit_compute_dtype=(
            torch.float16
        ),
    )


    # -------------------------------------------------------------------------
    # Step 8：載入 4-bit base model
    # -------------------------------------------------------------------------

    # 這裡不直接把 MODEL_ID 字串交給 SFTTrainer，
    # 而是自行先載入模型。
    #
    # 好處：
    # 可以非常清楚控制：
    # - quantization
    # - dtype
    # - device placement
    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_ID,

            # 套用上面定義的 4-bit 設定。
            quantization_config=(
                quant_config
            ),

            # 非量化部分使用 FP16。
            #
            # 注意：
            # 4-bit weights 本身仍是 quantized representation。
            #
            # Transformers 新版已將 torch_dtype 參數改名為 dtype，
            # 因此這裡直接使用 dtype，避免：
            #
            #     `torch_dtype` is deprecated! Use `dtype` instead!
            dtype=torch.float16,

            # {"": 0}：
            # 把整個模型放到 CUDA device 0。
            #
            # Colab 通常只有一張 GPU，
            # 所以 device 0 就是 T4。
            device_map={"": 0},

            trust_remote_code=(
                TRUST_REMOTE_CODE
            ),
        )
    )


    # -------------------------------------------------------------------------
    # Step 9：關閉 KV cache
    # -------------------------------------------------------------------------

    # KV cache 主要是「推論」時加速 autoregressive generation。
    #
    # 訓練時使用 gradient checkpointing，
    # use_cache=True 可能造成衝突或額外記憶體消耗。
    #
    # 所以 training 時關掉。
    model.config.use_cache = False


    # -------------------------------------------------------------------------
    # Step 10：準備 k-bit model 進行訓練
    # -------------------------------------------------------------------------

    # prepare_model_for_kbit_training()
    # 是 PEFT 在量化模型微調中的標準步驟。
    #
    # 它會做一些必要處理，例如：
    # - 凍結 base model parameters
    # - 處理 layer normalization precision
    # - 配合 gradient checkpointing
    #
    # 目的是讓：
    #
    # quantized base model
    #
    # 可以安全地搭配 LoRA adapter 訓練。
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=(
            GRADIENT_CHECKPOINTING
        ),
    )


    # -------------------------------------------------------------------------
    # Step 11：建立 LoRA 設定
    # -------------------------------------------------------------------------

    lora_config = LoraConfig(

        # 低秩矩陣的 rank。
        r=LORA_R,

        # LoRA scaling 參數。
        lora_alpha=LORA_ALPHA,

        # LoRA path 的 dropout。
        lora_dropout=LORA_DROPOUT,

        # 對所有 linear layers 套用 LoRA。
        target_modules=(
            LORA_TARGET_MODULES
        ),

        # 這是 causal language model。
        task_type="CAUSAL_LM",
    )


    # -------------------------------------------------------------------------
    # Step 12：把 LoRA adapter 加到 base model
    # -------------------------------------------------------------------------

    # get_peft_model() 之後，
    # model 仍然包含原本的 base model，
    # 但多出少量 LoRA adapter parameters。
    #
    # Base model：
    # requires_grad=False
    #
    # LoRA：
    # requires_grad=True
    model = get_peft_model(
        model,
        lora_config,
    )


    # -------------------------------------------------------------------------
    # Step 13：確保 LoRA trainable parameters 使用 FP32
    # -------------------------------------------------------------------------

    # 這是本程式針對 T4 的重要保護。
    #
    # Base model 仍是 4-bit quantized。
    #
    # 只有 LoRA 等少量 trainable parameters
    # 被保持為 FP32。
    cast_trainable_parameters_to_fp32(
        model
    )


    # -------------------------------------------------------------------------
    # Step 14：顯示 trainable parameter 數量
    # -------------------------------------------------------------------------

    # 你可能看到類似：
    #
    # trainable params: 4,399,104
    # all params: 498,431,872
    # trainable%: 0.8826
    #
    # 這正是 LoRA 的核心：
    #
    # 只更新不到 1% 的參數，
    # 而不是全部 5 億參數。
    model.print_trainable_parameters()


    # -------------------------------------------------------------------------
    # Step 15：檢查 trainable parameter dtype
    # -------------------------------------------------------------------------

    # 理想輸出：
    #
    # Trainable parameter dtypes:
    #   torch.float32: ...
    #
    # 如果有 BF16，
    # 程式會直接停止。
    print_trainable_dtypes(model)


    # -------------------------------------------------------------------------
    # Step 16：建立 SFT 訓練設定
    # -------------------------------------------------------------------------

    args = SFTConfig(

        # checkpoint 與最後輸出的儲存位置。
        output_dir=str(output_dir),

        # 每筆序列最大 token 長度。
        max_length=MAX_LENGTH,

        # 最大 epoch 數。
        num_train_epochs=EPOCHS,

        # Optimizer 的 learning rate。
        learning_rate=LEARNING_RATE,

        # 每張 GPU 的 train batch size。
        per_device_train_batch_size=(
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),

        # 每張 GPU 的 validation batch size。
        per_device_eval_batch_size=(
            PER_DEVICE_EVAL_BATCH_SIZE
        ),

        # 累積多少 mini-batches
        # 才做一次 optimizer step。
        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        # 開啟 gradient checkpointing，
        # 以運算時間換 GPU memory。
        gradient_checkpointing=(
            GRADIENT_CHECKPOINTING
        ),

        # -------------------------------------------------------------
        # Mixed Precision
        # -------------------------------------------------------------

        # FP16 = 16-bit floating point。
        #
        # 相較 FP32：
        # - 記憶體較省
        # - GPU 計算通常更快
        #
        # T4 適合使用 FP16。
        fp16=True,

        # 明確禁用 BF16。
        #
        # 避免再次出現：
        #
        # _amp_foreach_non_finite_check_and_unscale_cuda
        # not implemented for 'BFloat16'
        bf16=False,

        # TF32 是 Ampere 之後 GPU 常見的加速格式。
        #
        # T4 不是 Ampere，
        # 因此這裡明確關閉。
        tf32=False,

        # 使用較省記憶體的 optimizer。
        optim=OPTIM,

        # 每幾個 optimizer steps 顯示 training log。
        logging_steps=LOGGING_STEPS,

        # Evaluation strategy。
        eval_strategy=CHECK_STRATEGY,

        # 每隔幾個 steps 做 evaluation。
        eval_steps=CHECK_STEPS,

        # Checkpoint strategy。
        save_strategy=CHECK_STRATEGY,

        # 每隔幾個 steps 存 checkpoint。
        save_steps=CHECK_STEPS,

        # 最多保留幾份 checkpoint。
        save_total_limit=(
            CHECKPOINT_LIMIT
        ),

        # 訓練結束後，
        # 自動把 validation 指標最好的 checkpoint
        # 載回記憶體。
        load_best_model_at_end=True,

        # 用 eval_loss 判斷最佳模型。
        metric_for_best_model=(
            "eval_loss"
        ),

        # eval_loss 越低越好。
        #
        # 如果是 accuracy，
        # 通常才會設成 True。
        greater_is_better=False,

        # 只對 assistant completion 計算 loss。
        completion_only_loss=(
            COMPLETION_ONLY_LOSS
        ),

        # 指定 Qwen chat EOS token。
        eos_token=EOS_TOKEN,

        # 不使用 wandb 等外部 tracker。
        report_to=REPORT_TO,

        # 從 checkpoint resume 時，
        # 一併恢復 callback state。
        #
        # 對 early stopping 特別重要，
        # 不然 patience counter 可能重新開始計算。
        restore_callback_states_from_checkpoint=True,

        # 固定 random seed。
        seed=SEED,
    )


    # -------------------------------------------------------------------------
    # Step 17：建立 SFTTrainer
    # -------------------------------------------------------------------------

    trainer = SFTTrainer(

        # model 已經提前完成：
        #
        # 1. 4-bit quantization
        # 2. prepare_model_for_kbit_training
        # 3. LoRA injection
        #
        # 所以這裡直接傳 model object。
        model=model,

        # 上面建立的 training configuration。
        args=args,

        # 真正拿來計算 gradient 的 training split。
        train_dataset=data["train"],

        # 只用來做 evaluation 的 validation split。
        eval_dataset=data["validation"],

        # tokenizer / processor。
        processing_class=tokenizer,

        # EarlyStoppingCallback：
        #
        # 如果 eval_loss 連續多次不再改善，
        # 自動結束 training。
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=(
                    EARLY_STOPPING_PATIENCE
                )
            )
        ],

        # 注意：
        #
        # 這裡「故意不再」傳：
        #
        # quantization_config=...
        # peft_config=...
        #
        # 因為前面已經手動完成：
        #
        # quantization + LoRA
        #
        # 如果再傳一次，
        # 可能導致重複處理。
    )


    # -------------------------------------------------------------------------
    # Step 18：Trainer 建立後，把 LoRA trainable parameters 再轉回 FP32
    # -------------------------------------------------------------------------

    # 這一步是本程式在新版 TRL + QLoRA + Tesla T4 上的重要相容性修正。
    #
    # 為什麼前面已經轉成 FP32，這裡還要再做一次？
    #
    # 因為某些新版 TRL 的 Trainer 在初始化「量化模型 + PEFT」
    # 時，會依照 QLoRA 的 BF16 建議，把 requires_grad=True 的
    # PEFT adapter parameters 重新轉成 torch.bfloat16。
    #
    # 這正是我們實際觀察到的情況：
    #
    # 建立 SFTTrainer 之前：
    #
    #     Trainable parameter dtypes:
    #       torch.float32: 336
    #
    # 建立 SFTTrainer 之後：
    #
    #     Trainable parameter dtypes:
    #       torch.bfloat16: 336
    #
    # 對 Ampere / Ada / Hopper 等原生支援 BF16 的 GPU，
    # BF16 QLoRA 可以是合理選擇。
    #
    # 但是 Tesla T4 是 compute capability 7.5，
    # 沒有原生 BF16 Tensor Core。
    #
    # 更重要的是，我們的 Trainer 設定是：
    #
    #     fp16=True
    #     bf16=False
    #
    # 這會啟用 FP16 AMP GradScaler。
    #
    # 如果 trainable gradients 卻是 BF16，
    # GradScaler 在 unscale 階段可能出現：
    #
    #     NotImplementedError:
    #     "_amp_foreach_non_finite_check_and_unscale_cuda"
    #     not implemented for 'BFloat16'
    #
    # 所以在 SFTTrainer 完成初始化之後，
    # 必須把少量 LoRA trainable parameters 再轉回 FP32。
    #
    # 注意：
    # 此時 optimizer 尚未開始真正執行訓練更新，
    # 因此在這裡修正 parameter dtype 是合適的時機。
    #
    # Base model 仍然維持 4-bit quantized；
    # 不會因為這一行而把整個模型轉成 FP32。
    cast_trainable_parameters_to_fp32(
        trainer.model
    )


    # -------------------------------------------------------------------------
    # Step 19：Trainer 建立後，做最後一次 dtype 檢查
    # -------------------------------------------------------------------------

    # 現在理想輸出應該再次回到：
    #
    #     Trainable parameter dtypes:
    #       torch.float32: 336
    #
    # 如果仍有 BF16，程式會直接停止，
    # 避免等到 optimizer step 才出現難懂的 AMP 錯誤。
    print_trainable_dtypes(
        trainer.model
    )


    # -------------------------------------------------------------------------
    # Step 20：開始訓練
    # -------------------------------------------------------------------------

    # trainer.train() 內部會反覆執行：
    #
    # 1. 取一個 batch
    # 2. forward
    # 3. 計算 loss
    # 4. backward
    # 5. gradient accumulation
    # 6. optimizer.step()
    # 7. scheduler.step()
    # 8. 必要時 evaluation
    # 9. 必要時 checkpoint
    #
    # resume_from_checkpoint=None：
    #   從頭開始。
    #
    # resume_from_checkpoint=True / path：
    #   從既有 checkpoint 繼續。
    trainer.train(
        resume_from_checkpoint=(
            resume_from_checkpoint
        )
    )


    # -------------------------------------------------------------------------
    # Step 21：顯示最佳 checkpoint
    # -------------------------------------------------------------------------

    # 例如：
    #
    # outputs/qwen25_05b_lora/checkpoint-250
    print(
        "最佳 checkpoint："
        f"{trainer.state.best_model_checkpoint}"
    )

    # 例如：
    #
    # 最佳 eval_loss：0.8421
    print(
        "最佳 eval_loss："
        f"{trainer.state.best_metric}"
    )


    # -------------------------------------------------------------------------
    # Step 22：儲存最佳模型
    # -------------------------------------------------------------------------

    # 因為：
    #
    # load_best_model_at_end=True
    #
    # 所以此時 trainer.model 已經重新載入
    # validation 表現最好的 checkpoint。
    #
    # QLoRA / PEFT 情況下，
    # save_model() 主要會儲存 LoRA adapter，
    # 而不是重新複製完整 base model。
    trainer.save_model(
        str(output_dir)
    )


    # -------------------------------------------------------------------------
    # Step 23：儲存 Trainer state
    # -------------------------------------------------------------------------

    # Trainer state 包含例如：
    #
    # - global_step
    # - log_history
    # - best_metric
    # - best_model_checkpoint
    #
    # 對後續分析與續訓很有幫助。
    trainer.save_state()


    # -------------------------------------------------------------------------
    # Step 24：儲存 tokenizer
    # -------------------------------------------------------------------------

    # 推論時通常需要：
    #
    # Base model
    # + LoRA adapter
    # + Tokenizer
    #
    # 所以 tokenizer 也一起存進 OUTPUT_DIR。
    tokenizer.save_pretrained(
        str(output_dir)
    )


# =============================================================================
# 6. Python 程式進入點
# =============================================================================


# __name__ == "__main__"
#
# 表示這個檔案是直接執行：
#
#     python train_qlora.py
#
# 才會執行 main()。
#
# 如果其他 Python 程式只是：
#
#     import train_qlora
#
# 就不會立刻開始訓練。
#
# 這是 Python 專案中很常見的標準寫法。
if __name__ == "__main__":
    main()
