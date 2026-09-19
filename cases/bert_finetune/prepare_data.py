"""
prepare_data.py

把原始 reviews.txt 轉成 BERT 二元分類訓練所需的 train / validation JSONL。

原始資料格式是一行一筆：

    "評論文字"    1
    "評論文字"    0

中間使用 Tab（\t）分隔。

這支程式做的事情：

    reviews.txt
        ↓
    讀取文字與 label
        ↓
    檢查格式
        ↓
    以 label 做 stratified train / validation split
        ↓
    data/bert/train.jsonl
    data/bert/valid.jsonl

為什麼要先獨立做資料切分？

因為之後 train_bert.py 每次執行時，都直接讀取固定的 train / validation 檔案，
比較不同模型或超參數時，不會因為每次重新亂數切分而改變資料集。
"""

# =============================================================================
# 1. 匯入套件
# =============================================================================

# csv：
# 原始 reviews.txt 的文字欄位被雙引號包起來，而且使用 Tab 分隔。
# 使用 csv.reader 比手動 split("\t") 更穩定，也會自動移除外層雙引號。
import csv

# json：
# 用來把 Python dict 寫成 JSONL。
import json

# Counter：
# 用來統計 label 0 / 1 各有多少筆。
from collections import Counter

# Path：
# 用來處理輸入與輸出路徑。
from pathlib import Path

# train_test_split：
# 用來切 training / validation。
#
# 這裡使用 stratify=labels，
# 讓 train 與 validation 的 0 / 1 比例盡量接近原始資料。
from sklearn.model_selection import train_test_split


# =============================================================================
# 2. 資料設定
# =============================================================================

# 原始資料檔。
INPUT_FILE = Path("reviews.txt")

# 輸出資料夾。
OUTPUT_DIR = Path("data/bert")

# Validation 比例。
#
# 0.2 = 20%
#
# 所以大約：
#     80% → training
#     20% → validation
VALID_RATIO = 0.2

# 固定 random seed。
#
# 讓每次執行都得到相同的 train / validation split，
# 方便重現實驗。
SEED = 42


# =============================================================================
# 3. 輔助函式
# =============================================================================

def load_reviews(path: Path):
    """
    讀取 reviews.txt。

    回傳：
        texts  : list[str]
        labels : list[int]

    我們預期 label 只能是：
        0
        1
    """

    texts = []
    labels = []

    # 記錄被略過的空評論行號。
    #
    # 這些 row 雖然可能帶有 label，
    # 但沒有任何文字內容，
    # 對 text classification 沒有可學習的輸入資訊。
    skipped_empty_rows = []

    # newline="" 是 csv 模組官方建議的開檔方式，
    # 可以讓 csv.reader 正確處理換行。
    with path.open("r", encoding="utf-8", newline="") as file:

        # delimiter="\t"：
        #     使用 Tab 分隔欄位。
        #
        # quotechar='"'：
        #     文字欄位外層的雙引號由 csv.reader 處理，
        #     不會把雙引號本身留下來當成評論文字。
        reader = csv.reader(
            file,
            delimiter="\t",
            quotechar='"',
        )

        for line_number, row in enumerate(reader, start=1):

            # 空白行直接略過。
            if not row:
                continue

            # 正常情況應該剛好有兩欄：
            #
            # row[0] = 評論文字
            # row[1] = label
            if len(row) != 2:
                raise ValueError(
                    f"{path} 第 {line_number} 行欄位數不是 2：{row}"
                )

            text = row[0].strip()

            # -------------------------------------------------------------
            # 空評論直接略過
            # -------------------------------------------------------------
            #
            # 原始 reviews.txt 中可能存在：
            #
            #     ""\t0
            #
            # 也就是：
            #
            #     評論文字 = 空字串
            #     label    = 0
            #
            # 這種資料沒有任何文字資訊可以讓 BERT 學習，
            # 因此不應該拿去做訓練。
            #
            # 與其讓整個資料前處理流程因一筆空資料中止，
            # 這裡選擇：
            #
            #     skip
            #
            # 並記錄被略過的行號，最後統一顯示統計資訊。
            if not text:
                skipped_empty_rows.append(line_number)
                continue

            try:
                label = int(row[1])
            except ValueError as error:
                raise ValueError(
                    f"{path} 第 {line_number} 行的 label 不是整數：{row[1]!r}"
                ) from error

            # 本專案是 binary classification。
            if label not in (0, 1):
                raise ValueError(
                    f"{path} 第 {line_number} 行的 label 必須是 0 或 1，"
                    f"實際為 {label}。"
                )

            texts.append(text)
            labels.append(label)

    if not texts:
        raise RuntimeError(f"{path} 沒有讀到任何有效資料。")

    # 顯示空評論清理結果。
    #
    # 這不是 training error，
    # 而是資料清理的一部分。
    if skipped_empty_rows:
        print(
            "已略過空評論："
            f"{len(skipped_empty_rows)} 筆；"
            f"原始行號={skipped_empty_rows}"
        )

    return texts, labels


def save_jsonl(texts, labels, path: Path):
    """
    將文字與標籤寫成 JSONL。

    每一行格式：

        {"sentences": "這間飯店很好", "labels": 1}

    為了和原 notebook 的欄位名稱保持一致，
    這裡繼續使用：
        sentences
        labels
    """

    with path.open("w", encoding="utf-8") as file:
        for text, label in zip(texts, labels):

            row = {
                "sentences": text,
                "labels": label,
            }

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def print_label_distribution(name, labels):
    """
    顯示 0 / 1 的資料量與比例。
    """

    counts = Counter(labels)
    total = len(labels)

    print(f"{name}：{total} 筆")

    for label in sorted(counts):
        count = counts[label]
        ratio = count / total

        print(
            f"  label={label}: "
            f"{count} 筆 "
            f"({ratio:.2%})"
        )


# =============================================================================
# 4. 主流程
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Step 1：確認原始資料存在
    # -------------------------------------------------------------------------

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"找不到原始資料：{INPUT_FILE.resolve()}"
        )

    # -------------------------------------------------------------------------
    # Step 2：讀取 reviews.txt
    # -------------------------------------------------------------------------

    texts, labels = load_reviews(INPUT_FILE)

    print("=" * 72)
    print_label_distribution("全部資料", labels)

    # -------------------------------------------------------------------------
    # Step 3：切 train / validation
    # -------------------------------------------------------------------------

    # stratify=labels 很重要。
    #
    # 你的資料不是完全 50 / 50，
    # 所以如果只做普通亂數切分，
    # train 與 validation 的 label 比例可能出現額外偏差。
    #
    # stratify 會讓兩邊盡量維持相似的 class distribution。
    (
        train_texts,
        valid_texts,
        train_labels,
        valid_labels,
    ) = train_test_split(
        texts,
        labels,
        test_size=VALID_RATIO,
        random_state=SEED,
        stratify=labels,
    )

    # -------------------------------------------------------------------------
    # Step 4：建立輸出資料夾
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = OUTPUT_DIR / "train.jsonl"
    valid_path = OUTPUT_DIR / "valid.jsonl"

    # -------------------------------------------------------------------------
    # Step 5：儲存 JSONL
    # -------------------------------------------------------------------------

    save_jsonl(
        train_texts,
        train_labels,
        train_path,
    )

    save_jsonl(
        valid_texts,
        valid_labels,
        valid_path,
    )

    # -------------------------------------------------------------------------
    # Step 6：顯示結果
    # -------------------------------------------------------------------------

    print_label_distribution(
        "Training",
        train_labels,
    )

    print_label_distribution(
        "Validation",
        valid_labels,
    )

    print(f"Training JSONL：{train_path}")
    print(f"Validation JSONL：{valid_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
