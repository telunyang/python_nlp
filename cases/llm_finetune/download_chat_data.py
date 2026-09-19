"""
下載 Hugging Face 上的繁體中文多輪對話資料集，
篩選符合條件的 conversation，最後輸出成統一格式的 JSONL。

這個程式主要負責「原始資料取得與基本整理」。

整體流程可以先理解成：

    Hugging Face Dataset
        ↓
    下載指定 split
        ↓
    shuffle 打亂順序
        ↓
    逐筆讀取 conversation
        ↓
    過濾太短的對話
        ↓
    統一包成 {"messages": [...]} 格式
        ↓
    寫入 data/chat.jsonl
        ↓
    交給後續 prepare_chat_data.py 使用

這裡輸出的每一行都是一整段完整對話，例如：

    {"messages": [
        {"role": "user", "content": "我要一杯冰拿鐵"},
        {"role": "assistant", "content": "好的，一杯大杯冰拿鐵。"},
        {"role": "user", "content": "幫我加一份濃縮"},
        {"role": "assistant", "content": "好的，已幫您加一份濃縮咖啡。"}
    ]}

為什麼使用 JSONL，而不是一般 JSON？

JSONL = JSON Lines。

一般 JSON 可能會長這樣：

    [
        {...},
        {...},
        {...}
    ]

JSONL 則是：

    {...}
    {...}
    {...}

也就是「一行一筆 JSON」。

這樣的好處包括：
1. 可以逐行讀取，不需要一次把全部資料載入記憶體。
2. 大型語料處理比較方便。
3. Hugging Face Datasets、pandas、Python 都很容易處理。
4. 某一行壞掉時，比較容易定位問題。

這個資料集本身已經提供 conversations / role / content 結構，
所以這支程式不需要做複雜的欄位轉換，只需要：
1. 下載。
2. 打亂順序。
3. 過濾太短的 conversation。
4. 控制最多保留多少筆。
5. 輸出成統一 JSONL。
"""

# =============================================================================
# 1. 匯入 Python 標準函式庫
# =============================================================================

# json 是 Python 內建模組。
#
# 這裡主要用 json.dumps()，
# 把 Python dict / list 轉成 JSON 字串。
#
# 例如：
#
# Python：
#     {"role": "user", "content": "你好"}
#
# 經過：
#     json.dumps(...)
#
# 會變成 JSON 字串：
#     {"role": "user", "content": "你好"}
#
# 我們會把每一筆 conversation 寫成一行 JSONL。
import json


# pathlib.Path 是 Python 內建的路徑處理工具。
#
# 相較於直接拼接字串：
#
#     "data/" + "chat.jsonl"
#
# Path 寫法：
#
#     Path("data/chat.jsonl")
#
# 比較清楚，而且可以直接：
# - 建立資料夾
# - 檢查檔案是否存在
# - 開啟檔案
from pathlib import Path


# =============================================================================
# 2. 匯入 Hugging Face Datasets
# =============================================================================

# load_dataset() 可以直接從 Hugging Face Hub 下載公開資料集。
#
# 例如：
#
#     load_dataset("renhehuang/coffee-order-zhtw", split="train")
#
# Hugging Face Datasets 會自動處理：
# - 下載
# - cache
# - schema
# - split
#
# 回傳的是 Dataset object，
# 可以像 Python list 一樣逐筆迭代。
from datasets import load_dataset


# =============================================================================
# 3. 語料設定
# =============================================================================

# Hugging Face Hub 上的資料集 ID。
#
# 格式通常是：
#
#     使用者名稱 / 資料集名稱
#
# 例如：
#
#     renhehuang/coffee-order-zhtw
#
# 這個資料集主題集中在繁體中文咖啡點餐，
# 對課堂示範多輪對話 SFT 很適合，
# 因為任務範圍單純，不容易混入太多其他領域知識。
DATASET_ID = "renhehuang/coffee-order-zhtw"


# 指定要使用哪一個 dataset split。
#
# 常見 split 名稱：
#
#     train
#     validation
#     test
#
# 這個資料集主要提供 train split，
# 所以這裡使用 "train"。
SPLIT = "train"


# 最多輸出多少段完整 conversation。
#
# 這裡是「conversation 數量」，
# 不是 message 數量。
#
# 例如：
#
# 一段 conversation：
#
#     user
#     assistant
#     user
#     assistant
#
# 雖然有 4 則 message，
# 但只算 1 段 conversation。
#
# MAX_CONVERSATIONS = 2500
#
# 表示最多輸出 2,500 段完整多輪對話。
#
# 如果只是先測試 pipeline，
# 可以改小：
#
#     MAX_CONVERSATIONS = 100
#
# 或：
#
#     MAX_CONVERSATIONS = 500
#
# 這樣可以更快確認：
# - 資料格式正不正確
# - prepare_chat_data.py 是否能正常執行
# - train_qlora.py 是否能正常開始訓練
MAX_CONVERSATIONS = 2500


# 每一段 conversation 至少要有多少則 message。
#
# MIN_MESSAGES = 4
#
# 通常代表至少：
#
#     user
#     assistant
#     user
#     assistant
#
# 也就是至少 2 輪 user-assistant 對話。
#
# 為什麼不要只保留 2 則 message？
#
# 因為：
#
#     user
#     assistant
#
# 比較接近單輪問答。
#
# 本專案希望模型學到「多輪上下文」，
# 所以至少保留 4 則 message 比較合理。
MIN_MESSAGES = 4


# 最後輸出的 JSONL 路徑。
#
# 執行完成後會產生：
#
#     data/chat.jsonl
#
# 後續 prepare_chat_data.py 會讀取這個檔案，
# 再進一步：
# - 切 train / validation
# - 套用 chat template
# - 建立 prompt / completion
# - 做訓練前格式整理
OUTPUT_FILE = Path("data/chat.jsonl")


# Random seed。
#
# shuffle() 會打亂資料順序。
#
# 如果每次執行都不固定 seed，
# 每次前 2500 筆可能都不一樣。
#
# 固定：
#
#     SEED = 42
#
# 可以讓 shuffle 結果較容易重現。
#
# 注意：
# 固定 seed 的目的不是讓「模型完全 deterministic」，
# 而是讓這個資料抽樣流程保持一致。
SEED = 42


# =============================================================================
# 4. 主流程
# =============================================================================

def main():
    """
    執行資料下載、shuffle、篩選與 JSONL 輸出。

    流程：

    1. 從 Hugging Face Hub 下載資料。
    2. 對 dataset 做 shuffle。
    3. 建立輸出資料夾。
    4. 逐筆讀取 conversation。
    5. 過濾太短的對話。
    6. 將 conversation 統一包成 {"messages": ...}。
    7. 一行一筆寫入 JSONL。
    8. 達到 MAX_CONVERSATIONS 後停止。
    """

    # -------------------------------------------------------------------------
    # Step 1：下載資料集
    # -------------------------------------------------------------------------

    # load_dataset() 會：
    #
    # 1. 先檢查本機 Hugging Face cache。
    # 2. 如果沒有，再從 Hugging Face Hub 下載。
    # 3. 讀取指定 split。
    #
    # 回傳的 dataset 可以像這樣使用：
    #
    #     dataset[0]
    #
    # 取得第一筆資料。
    dataset = load_dataset(
        DATASET_ID,
        split=SPLIT,
    )


    # -------------------------------------------------------------------------
    # Step 2：打亂資料順序
    # -------------------------------------------------------------------------

    # 為什麼要 shuffle？
    #
    # 因為如果原始資料集有某種排序，
    # 例如：
    # - 類似案例放在一起
    # - 簡單樣本排前面
    # - 特定模板集中
    #
    # 如果我們直接取前 2500 筆，
    # 可能會產生抽樣偏差。
    #
    # shuffle(seed=SEED) 讓資料先被重新排列。
    dataset = dataset.shuffle(
        seed=SEED
    )


    # -------------------------------------------------------------------------
    # Step 3：建立輸出資料夾
    # -------------------------------------------------------------------------

    # OUTPUT_FILE：
    #
    #     data/chat.jsonl
    #
    # OUTPUT_FILE.parent：
    #
    #     data
    #
    # mkdir() 會建立資料夾。
    #
    # parents=True：
    # 如果上層資料夾也不存在，一併建立。
    #
    # exist_ok=True：
    # 如果 data/ 已經存在，不要報錯。
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -------------------------------------------------------------------------
    # Step 4：初始化計數器
    # -------------------------------------------------------------------------

    # count 用來記錄：
    #
    # 「目前已經成功寫入多少段 conversation」
    #
    # 注意：
    # 它不是目前讀過多少 row。
    #
    # 因為某些 row 可能因為太短而被 skip。
    count = 0


    # -------------------------------------------------------------------------
    # Step 5：開啟 JSONL 輸出檔
    # -------------------------------------------------------------------------

    # mode="w"：
    #
    # 每次執行都重新建立檔案。
    #
    # 如果原本已有 data/chat.jsonl，
    # 會被覆蓋。
    #
    # encoding="utf-8"：
    #
    # 確保繁體中文可以正確寫入。
    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        # ---------------------------------------------------------------------
        # Step 6：逐筆處理 dataset
        # ---------------------------------------------------------------------

        # row 代表一筆原始 dataset record。
        #
        # 例如 row 可能像：
        #
        # {
        #     "conversations": [
        #         {"role": "user", "content": "..."},
        #         {"role": "assistant", "content": "..."}
        #     ]
        # }
        for row in dataset:

            # -----------------------------------------------------------------
            # Step 7：取出 conversations 欄位
            # -----------------------------------------------------------------

            # 原始資料集已經使用 conversations 欄位，
            # 所以不需要額外把 from / value 等其他 schema
            # 轉換成 role / content。
            messages = row["conversations"]


            # -----------------------------------------------------------------
            # Step 8：過濾太短的 conversation
            # -----------------------------------------------------------------

            # len(messages) 代表這段 conversation 有幾則 message。
            #
            # 例如：
            #
            # [
            #     user,
            #     assistant,
            #     user,
            #     assistant
            # ]
            #
            # len = 4
            #
            # 如果少於 MIN_MESSAGES，
            # 就不符合多輪對話需求。
            if len(messages) < MIN_MESSAGES:

                # continue：
                #
                # 立刻結束這一輪 for loop，
                # 跳到下一筆 row。
                continue


            # -----------------------------------------------------------------
            # Step 9：統一輸出格式
            # -----------------------------------------------------------------

            # 我們希望最後每一行都長成：
            #
            # {
            #     "messages": [
            #         ...
            #     ]
            # }
            #
            # 即使原始資料欄位名稱是 conversations，
            # 輸出時統一改成 messages，
            # 讓後續程式只需要處理單一格式。
            output_row = {
                "messages": messages
            }


            # -----------------------------------------------------------------
            # Step 10：Python dict → JSON 字串
            # -----------------------------------------------------------------

            # json.dumps() 把 Python object 轉成 JSON string。
            #
            # ensure_ascii=False 很重要。
            #
            # 如果使用預設 ensure_ascii=True，
            # 中文可能被寫成：
            #
            #     "\u4f60\u597d"
            #
            # 雖然技術上仍是合法 JSON，
            # 但人類閱讀很不方便。
            #
            # 使用 False 後會直接保留：
            #
            #     "你好"
            json_line = json.dumps(
                output_row,
                ensure_ascii=False,
            )


            # -----------------------------------------------------------------
            # Step 11：寫入一行 JSONL
            # -----------------------------------------------------------------

            # JSONL 的核心規則：
            #
            # 一行 = 一筆 JSON。
            #
            # 所以每次寫完一筆都要補：
            #
            #     "\n"
            #
            # 讓下一筆從新的一行開始。
            file.write(
                json_line + "\n"
            )


            # -----------------------------------------------------------------
            # Step 12：增加成功輸出數量
            # -----------------------------------------------------------------

            count += 1


            # -----------------------------------------------------------------
            # Step 13：達到最大資料量後停止
            # -----------------------------------------------------------------

            # 假設：
            #
            # MAX_CONVERSATIONS = 2500
            #
            # 當 count == 2500，
            # 就停止讀取剩下資料。
            #
            # 這可以控制：
            # - 訓練時間
            # - 資料量
            # - 課堂示範成本
            if count >= MAX_CONVERSATIONS:

                # break：
                #
                # 直接離開 for row in dataset 迴圈。
                break


    # -------------------------------------------------------------------------
    # Step 14：顯示完成訊息
    # -------------------------------------------------------------------------

    print(
        f"已輸出 {count} 段對話："
        f"{OUTPUT_FILE}"
    )


# =============================================================================
# 5. Python 程式進入點
# =============================================================================

# __name__ == "__main__"
#
# 表示只有在使用：
#
#     python download_chat_data.py
#
# 直接執行這個檔案時，
# 才會執行 main()。
#
# 如果其他程式只是：
#
#     import download_chat_data
#
# 則不會自動開始下載資料。
#
# 這是 Python 專案中很常見的標準結構。
if __name__ == "__main__":

    # 執行主流程。
    main()
