"""
把完整的多輪 conversation，展開成適合 TRL Supervised Fine-Tuning（SFT）
使用的 prompt / completion JSONL。

這支程式位在整個資料處理流程的「第二階段」。

完整流程可以先理解成：

    Hugging Face Dataset
        ↓
    download_chat_data.py
        ↓
    data/chat.jsonl
        ↓
    prepare_chat_data.py   ← 這支程式
        ↓
    data/sft/train.jsonl
    data/sft/valid.jsonl
        ↓
    train_qlora.py
        ↓
    QLoRA 微調

------------------------------------------------------------
這支程式主要做 4 件事
------------------------------------------------------------

1. 讀取完整 conversation。
2. 先用「完整 conversation」切 train / validation。
3. 再把每段 conversation 展開成多筆：
       prompt → completion
4. 輸出成 TRL 可直接讀取的 conversational JSONL。

------------------------------------------------------------
為什麼不能先把每一個 turn 展開，再切 train / validation？
------------------------------------------------------------

假設原始 conversation 是：

    user: 我想點一杯冰拿鐵
    assistant: 好的，一杯大杯冰拿鐵
    user: 幫我加一份濃縮
    assistant: 好的，已幫您加一份濃縮咖啡
    user: 再來一杯熱的
    assistant: 好的，再加一杯大杯熱拿鐵

如果直接先展開，可能得到：

    Example A
    prompt:
        user: 我想點一杯冰拿鐵
    completion:
        assistant: 好的，一杯大杯冰拿鐵

    Example B
    prompt:
        user: 我想點一杯冰拿鐵
        assistant: 好的，一杯大杯冰拿鐵
        user: 幫我加一份濃縮
    completion:
        assistant: 好的，已幫您加一份濃縮咖啡

    Example C
    prompt:
        ...
    completion:
        assistant: 好的，再加一杯大杯熱拿鐵

如果 A 被分到 train，
B 或 C 被分到 validation，
那麼 train 與 validation 其實來自「同一段原始對話」。

這會造成 turn-level leakage：

    train 看過：
        「我想點一杯冰拿鐵」

    validation 又出現：
        「我想點一杯冰拿鐵」
        「好的，一杯大杯冰拿鐵」
        ...

模型會因為已經在 train 看過高度相似的上下文，
導致 validation loss 看起來過度樂觀。

因此正確順序應該是：

    完整 conversation
        ↓
    先切 train / validation
        ↓
    再各自展開

這樣可以確保同一段 conversation 的所有 turns，
只會存在於 train 或 validation 其中一邊。

------------------------------------------------------------
輸出的 TRL conversational 格式
------------------------------------------------------------

每一行會長得像：

    {
        "prompt": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "我想點一杯冰拿鐵"},
            {"role": "assistant", "content": "好的，一杯大杯冰拿鐵"},
            {"role": "user", "content": "幫我加一份濃縮"}
        ],
        "completion": [
            {"role": "assistant", "content": "好的，已幫您加一份濃縮咖啡"}
        ]
    }

也就是：

    prompt
        =
        模型在回答前可以看到的上下文

    completion
        =
        這一筆訓練資料希望模型學會產生的 assistant 回覆

如果 train_qlora.py 使用：

    completion_only_loss=True

那麼 loss 主要會針對 completion，
也就是 assistant 的回答部分計算。
"""

# =============================================================================
# 1. 匯入 Python 標準函式庫
# =============================================================================

# json 是 Python 內建模組。
#
# 在這支程式裡會同時用到：
#
# json.loads()
#     把 JSON 字串轉回 Python dict / list。
#
# json.dumps()
#     把 Python dict / list 轉成 JSON 字串。
#
# 因為輸入與輸出都是 JSONL，
# 所以每一行都需要各自做一次 parse / serialize。
import json


# random 是 Python 內建的亂數模組。
#
# 這裡不是拿來做模型訓練，
# 而是拿來打亂 conversation 順序，
# 再切 train / validation。
#
# 我們會使用：
#
#     random.Random(SEED)
#
# 而不是直接使用全域 random，
# 這樣可以讓這支程式自己的 shuffle 更容易重現。
import random


# pathlib.Path 是 Python 內建的路徑處理工具。
#
# 它比直接拼接字串更清楚。
#
# 例如：
#
#     Path("data/sft") / "train.jsonl"
#
# 會得到：
#
#     data/sft/train.jsonl
#
# 同時也可以直接：
# - 建立資料夾
# - 檢查檔案是否存在
# - 開啟檔案
from pathlib import Path


# =============================================================================
# 2. 資料設定
# =============================================================================

# download_chat_data.py 產生的完整 conversation JSONL。
#
# 預期檔案：
#
#     data/chat.jsonl
#
# 每一行代表一整段 conversation，例如：
#
# {
#   "messages": [
#     {"role": "user", "content": "..."},
#     {"role": "assistant", "content": "..."},
#     ...
#   ]
# }
#
# 注意：
# 這裡每一行還是「完整 conversation」，
# 尚未展開成 prompt / completion。
INPUT_FILE = Path("data/chat.jsonl")


# train / validation SFT JSONL 的輸出資料夾。
#
# 執行完後會產生：
#
#     data/sft/train.jsonl
#     data/sft/valid.jsonl
#
# train_qlora.py 會直接讀這兩個檔案。
OUTPUT_DIR = Path("data/sft")


# Validation conversation 的比例。
#
# VALID_RATIO = 0.1
#
# 代表：
#
#     90% conversation → train
#     10% conversation → validation
#
# 例如：
#
#     總共有 2500 段 conversation
#
# 那麼大約：
#
#     2250 段 → train
#      250 段 → validation
#
# 這裡切的是「conversation 數量」，
# 不是最後展開後的 prompt/completion example 數量。
VALID_RATIO = 0.1


# 每一筆 prompt 最多保留最近多少則「非 system message」。
#
# 這裡的 message 指：
#
#     user
#     assistant
#
# 例如：
#
# MAX_CONTEXT_MESSAGES = 6
#
# 代表最多保留最近 6 則非 system 訊息。
#
# 一段規則交錯的 conversation 可能是：
#
#     user       ← 1
#     assistant  ← 2
#     user       ← 3
#     assistant  ← 4
#     user       ← 5
#     assistant  ← 6
#
# 這大約相當於最近 3 輪 user-assistant 對話。
#
# 為什麼要截斷？
#
# 因為 context 越長：
# - token 數越多
# - GPU 記憶體用量越高
# - attention 計算量越大
# - T4 訓練速度越慢
#
# 本專案是咖啡點餐，
# 任務通常不需要保留非常長的歷史。
MAX_CONTEXT_MESSAGES = 6


# Random seed。
#
# 這裡控制的是：
#
#     conversation shuffle
#     train / validation split
#
# 固定：
#
#     SEED = 42
#
# 可以讓每次執行時：
#
#     哪些 conversation 被分到 validation
#
# 基本保持一致。
#
# 這對比較不同訓練設定非常重要。
#
# 否則如果每次 validation 集都不同，
# eval_loss 就不容易公平比較。
SEED = 42


# =============================================================================
# 3. 將完整 conversation 展開成 SFT examples
# =============================================================================

def build_examples(conversations):
    """
    將多段完整 conversation，
    展開成多筆「歷史上下文 → 下一個 assistant 回覆」。

    參數
    ----
    conversations:
        Python list。

        每一個元素都是一整段 messages，例如：

        [
            [
                {"role": "user", "content": "我要冰拿鐵"},
                {"role": "assistant", "content": "好的"},
                {"role": "user", "content": "加濃縮"},
                {"role": "assistant", "content": "沒問題"}
            ],

            [
                ...
            ]
        ]

    回傳
    ----
    examples:
        每一筆都是：

        {
            "prompt": [...],
            "completion": [...]
        }

    ------------------------------------------------------------
    展開概念
    ------------------------------------------------------------

    假設一段 conversation：

        user: U1
        assistant: A1
        user: U2
        assistant: A2
        user: U3
        assistant: A3

    會展開成：

        Example 1
        prompt:
            U1
        completion:
            A1

        Example 2
        prompt:
            U1
            A1
            U2
        completion:
            A2

        Example 3
        prompt:
            U1
            A1
            U2
            A2
            U3
        completion:
            A3

    也就是：
    每一個 assistant message，
    都可以成為一個 supervision target。
    """

    # 建立最後要回傳的 example 清單。
    #
    # 每找到一個合法的 assistant 回覆，
    # 就 append 一筆 prompt / completion。
    examples = []


    # -------------------------------------------------------------------------
    # Step 1：逐段處理 conversation
    # -------------------------------------------------------------------------

    for messages in conversations:

        # messages 是單一 conversation。
        #
        # 例如：
        #
        # [
        #     {"role": "system", ...},
        #     {"role": "user", ...},
        #     {"role": "assistant", ...},
        #     ...
        # ]


        # ---------------------------------------------------------------------
        # Step 2：逐則尋找 assistant message
        # ---------------------------------------------------------------------

        # enumerate(messages) 會同時得到：
        #
        # index
        #     訊息在 conversation 中的位置
        #
        # message
        #     該位置的訊息 dict
        #
        # 例如：
        #
        # index = 2
        # message = {
        #     "role": "assistant",
        #     "content": "好的"
        # }
        for index, message in enumerate(messages):


            # -----------------------------------------------------------------
            # Step 3：只把 assistant message 當成 completion
            # -----------------------------------------------------------------

            # 我們要訓練的是：
            #
            #     給定前面的 context
            #     預測 assistant 回覆
            #
            # 所以：
            #
            # role=user
            # role=system
            #
            # 都不能當 completion。
            #
            # index == 0 也直接跳過，
            # 因為第一則 message 前面沒有任何 context。
            if (
                index == 0
                or message["role"] != "assistant"
            ):
                continue


            # -----------------------------------------------------------------
            # Step 4：取出目前 assistant 回覆之前的全部 context
            # -----------------------------------------------------------------

            # messages[:index]
            #
            # Python slicing 的意思是：
            #
            # 從最前面開始，
            # 一直到 index 之前，
            # 不包含目前這個 assistant message。
            #
            # 例如：
            #
            # messages =
            #   0 user U1
            #   1 assistant A1
            #   2 user U2
            #   3 assistant A2
            #
            # 當 index = 3：
            #
            # context = messages[:3]
            #
            # 會得到：
            #
            #   user U1
            #   assistant A1
            #   user U2
            context = messages[:index]


            # -----------------------------------------------------------------
            # Step 5：確認 assistant 前一則是 user
            # -----------------------------------------------------------------

            # 標準 chat supervision 通常期待：
            #
            #     ... user
            #     assistant
            #
            # 如果 context 最後一則不是 user，
            # 例如：
            #
            #     assistant
            #     assistant
            #
            # 或：
            #
            #     system
            #     assistant
            #
            # 就不是我們希望的標準 user → assistant pair。
            if context[-1]["role"] != "user":
                continue


            # -----------------------------------------------------------------
            # Step 6：如果第一則是 system，就固定保留
            # -----------------------------------------------------------------

            # System message 通常定義：
            # - 模型角色
            # - 回答規則
            # - 任務限制
            #
            # 例如：
            #
            #     你是一位咖啡點餐助理...
            #
            # 如果有 system message，
            # 即使後面要截斷歷史，
            # 通常也應該保留。
            #
            # context[:1]
            #
            # 會得到只包含第一筆 message 的 list。
            if context[0]["role"] == "system":
                system = context[:1]
            else:
                system = []


            # -----------------------------------------------------------------
            # Step 7：移除 system，再取最近 N 則歷史
            # -----------------------------------------------------------------

            # 先用 list comprehension 移除 system message：
            #
            #     item["role"] != "system"
            #
            # 然後：
            #
            #     [-MAX_CONTEXT_MESSAGES:]
            #
            # 只保留最後 N 則。
            #
            # 例如 MAX_CONTEXT_MESSAGES = 6，
            # 原本有 12 則歷史，
            # 就只留下最近 6 則。
            history = [
                item
                for item in context
                if item["role"] != "system"
            ][-MAX_CONTEXT_MESSAGES:]


            # -----------------------------------------------------------------
            # Step 8：確保截斷後從 user 開始
            # -----------------------------------------------------------------

            # 假設原本是：
            #
            #     user
            #     assistant
            #     user
            #     assistant
            #     user
            #     assistant
            #     user
            #
            # 如果只取最後 6 則，
            # 可能剛好變成：
            #
            #     assistant   ← 不完整
            #     user
            #     assistant
            #     user
            #     assistant
            #     user
            #
            # 這樣第一則是 assistant，
            # 代表 prompt 一開始就少掉前面的 user 問題，
            # 語意不完整。
            #
            # 所以只要第一則不是 user，
            # 就持續 pop(0) 移除最前面的 message，
            # 直到從 user 開始。
            while (
                history
                and history[0]["role"] != "user"
            ):
                history.pop(0)


            # -----------------------------------------------------------------
            # Step 9：建立一筆 TRL conversational example
            # -----------------------------------------------------------------

            # prompt：
            #
            #     system
            #       +
            #     最近多輪 history
            #
            # completion：
            #
            #     目前這一則 assistant message
            #
            # completion 一樣使用 list，
            # 因為 TRL conversational format
            # 會以 message list 表示。
            example = {
                "prompt": system + history,
                "completion": [message],
            }


            # -----------------------------------------------------------------
            # Step 10：加入輸出 example
            # -----------------------------------------------------------------

            examples.append(example)


    # -------------------------------------------------------------------------
    # Step 11：回傳所有展開完成的 examples
    # -------------------------------------------------------------------------

    return examples


# =============================================================================
# 4. 將 Python dict list 寫成 JSONL
# =============================================================================

def save_jsonl(rows, path):
    """
    將多筆 Python dict 寫成 UTF-8 JSONL。

    例如 rows：

        [
            {"prompt": [...], "completion": [...]},
            {"prompt": [...], "completion": [...]},
        ]

    會輸出：

        {...}
        {...}

    一行一筆 JSON。
    """

    # mode="w"：
    # 每次執行都重新建立檔案。
    #
    # 如果原本已有 train.jsonl，
    # 會被覆蓋。
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        # 逐筆寫入。
        for row in rows:

            # ensure_ascii=False：
            #
            # 讓繁體中文直接保留：
            #
            #     "你好"
            #
            # 而不是：
            #
            #     "\u4f60\u597d"
            json_line = json.dumps(
                row,
                ensure_ascii=False,
            )

            # JSONL：
            # 每筆 JSON 後面都要加換行。
            file.write(
                json_line + "\n"
            )


# =============================================================================
# 5. 主流程
# =============================================================================

def main():
    """
    執行完整資料準備流程：

    1. 讀取 data/chat.jsonl
    2. 建立完整 conversation list
    3. 以固定 seed shuffle
    4. 先切 train / validation conversation
    5. 各自展開成 prompt / completion
    6. 建立 data/sft
    7. 寫入 train.jsonl / valid.jsonl
    8. 顯示統計資訊
    """


    # -------------------------------------------------------------------------
    # Step 1：確認輸入檔存在
    # -------------------------------------------------------------------------

    # 如果 download_chat_data.py 還沒執行，
    # data/chat.jsonl 可能不存在。
    #
    # 比起等 open() 出現不明確錯誤，
    # 這裡先提供清楚訊息。
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"找不到輸入檔：{INPUT_FILE}\n"
            "請先執行 download_chat_data.py。"
        )


    # -------------------------------------------------------------------------
    # Step 2：建立 conversation 清單
    # -------------------------------------------------------------------------

    conversations = []


    # -------------------------------------------------------------------------
    # Step 3：讀取完整 conversation JSONL
    # -------------------------------------------------------------------------

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        # line_number 從 1 開始，
        # 若 JSON 格式壞掉時比較容易定位是哪一行。
        for line_number, line in enumerate(
            file,
            start=1,
        ):

            # strip() 去掉前後空白與換行。
            #
            # 如果整行只有空白，
            # 就不需要處理。
            if not line.strip():
                continue


            # -------------------------------------------------------------
            # Step 4：JSON 字串 → Python dict
            # -------------------------------------------------------------

            try:
                row = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{INPUT_FILE} 第 {line_number} 行 "
                    f"不是合法 JSON：{error}"
                ) from error


            # -------------------------------------------------------------
            # Step 5：確認有 messages 欄位
            # -------------------------------------------------------------

            if "messages" not in row:
                raise KeyError(
                    f"{INPUT_FILE} 第 {line_number} 行 "
                    "缺少 messages 欄位。"
                )


            # 只取完整 messages conversation。
            conversations.append(
                row["messages"]
            )


    # -------------------------------------------------------------------------
    # Step 6：確認資料不是空的
    # -------------------------------------------------------------------------

    if not conversations:
        raise RuntimeError(
            f"{INPUT_FILE} 沒有任何 conversation。"
        )


    # -------------------------------------------------------------------------
    # Step 7：使用固定 seed 打亂 conversation
    # -------------------------------------------------------------------------

    # 建立一個獨立 Random instance。
    #
    # 好處：
    # 不會修改 Python 全域 random state。
    rng = random.Random(SEED)

    # 原地打亂 conversations。
    rng.shuffle(conversations)


    # -------------------------------------------------------------------------
    # Step 8：計算 validation conversation 數量
    # -------------------------------------------------------------------------

    # 例如：
    #
    # len(conversations) = 2500
    # VALID_RATIO = 0.1
    #
    # int(2500 × 0.1)
    # = 250
    #
    # max(1, ...)
    #
    # 確保即使資料很少，
    # validation 至少保留 1 段 conversation。
    valid_size = max(
        1,
        int(
            len(conversations)
            * VALID_RATIO
        ),
    )


    # -------------------------------------------------------------------------
    # Step 9：先以完整 conversation 切 validation
    # -------------------------------------------------------------------------

    # 取 shuffle 後最前面的 valid_size 段。
    #
    # 最重要的地方是：
    #
    # 這時 conversation 還沒有被拆成 turn-level examples。
    valid_conversations = (
        conversations[:valid_size]
    )


    # -------------------------------------------------------------------------
    # Step 10：剩下 conversation 作為 training data
    # -------------------------------------------------------------------------

    train_conversations = (
        conversations[valid_size:]
    )


    # -------------------------------------------------------------------------
    # Step 11：展開 training conversations
    # -------------------------------------------------------------------------

    # 假設一段 conversation 有 3 個 assistant messages，
    # 那麼通常可以展開成 3 筆 training examples。
    #
    # 所以：
    #
    # conversation 數
    #
    # 通常會小於：
    #
    # 最後 train_rows 的數量。
    train_rows = build_examples(
        train_conversations
    )


    # -------------------------------------------------------------------------
    # Step 12：展開 validation conversations
    # -------------------------------------------------------------------------

    valid_rows = build_examples(
        valid_conversations
    )


    # -------------------------------------------------------------------------
    # Step 13：建立輸出資料夾
    # -------------------------------------------------------------------------

    # 如果：
    #
    #     data/sft/
    #
    # 不存在，就自動建立。
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -------------------------------------------------------------------------
    # Step 14：儲存 training JSONL
    # -------------------------------------------------------------------------

    train_path = (
        OUTPUT_DIR / "train.jsonl"
    )

    save_jsonl(
        train_rows,
        train_path,
    )


    # -------------------------------------------------------------------------
    # Step 15：儲存 validation JSONL
    # -------------------------------------------------------------------------

    valid_path = (
        OUTPUT_DIR / "valid.jsonl"
    )

    save_jsonl(
        valid_rows,
        valid_path,
    )


    # -------------------------------------------------------------------------
    # Step 16：顯示統計資訊
    # -------------------------------------------------------------------------

    print("=" * 72)

    print(
        f"總 conversation 數："
        f"{len(conversations)}"
    )

    print(
        f"Train conversations："
        f"{len(train_conversations)}"
    )

    print(
        f"Validation conversations："
        f"{len(valid_conversations)}"
    )

    print(
        f"Train SFT examples："
        f"{len(train_rows)}"
    )

    print(
        f"Validation SFT examples："
        f"{len(valid_rows)}"
    )

    print(
        f"Train JSONL："
        f"{train_path}"
    )

    print(
        f"Validation JSONL："
        f"{valid_path}"
    )

    print("=" * 72)


# =============================================================================
# 6. Python 程式進入點
# =============================================================================

# 只有直接執行：
#
#     python prepare_chat_data.py
#
# 才會執行 main()。
#
# 如果其他程式只是：
#
#     import prepare_chat_data
#
# 就不會自動開始資料切分與輸出。
if __name__ == "__main__":

    # 執行資料準備主流程。
    main()
