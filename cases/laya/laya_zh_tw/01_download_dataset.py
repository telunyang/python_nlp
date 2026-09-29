import json
from pathlib import Path

from datasets import load_dataset

# 這個資料集全部是臺灣繁體中文咖啡點餐對話。
# 如果之後換英文或其他語言，最先改的就是這個 DATASET_ID。
DATASET_ID = "renhehuang/coffee-order-zhtw"
SEED = 42
MAX_CONVERSATIONS = 800  # 教學先用 800 段；想用完整 2.94k 資料可改成 None。

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"


def save_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    print(f"下載資料集：{DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train").shuffle(seed=SEED)

    if MAX_CONVERSATIONS is not None:
        dataset = dataset.select(range(min(MAX_CONVERSATIONS, len(dataset))))

    # 以「整段對話」切分，避免同一段對話的不同 turn 同時出現在 train/test。
    n = len(dataset)
    n_train = int(n * 0.8)
    n_calibration = int(n * 0.1)

    train = dataset.select(range(0, n_train))
    calibration = dataset.select(range(n_train, n_train + n_calibration))
    test = dataset.select(range(n_train + n_calibration, n))

    save_jsonl(train, RAW_DIR / "train.jsonl")
    save_jsonl(calibration, RAW_DIR / "calibration.jsonl")
    save_jsonl(test, RAW_DIR / "test.jsonl")

    print(f"train 對話       : {len(train)}")
    print(f"calibration 對話 : {len(calibration)}")
    print(f"test 對話        : {len(test)}")
    print(f"已存到：{RAW_DIR}")


if __name__ == "__main__":
    main()
