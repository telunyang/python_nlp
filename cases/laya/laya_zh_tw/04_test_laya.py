import json
import time
from pathlib import Path

import numpy as np
import torch

import laya
from laya.common import ece_score

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "output" / "laya_coffee_zh_tw"
TEST_FILE = HERE / "data" / "test.jsonl"
MAX_BENCHMARK_DECISIONS = 300  # 想跑完整 test，可以改成 None。


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def external_question(row):
    return {
        "type": row["type"],
        "instructions": row["instructions"],
        "criteria": row.get("criteria"),
    }


def probabilities_from_result(row, answer):
    if row["type"] == "noul":
        p_true = float(answer["noul"])
        return np.array([1.0 - p_true, p_true], dtype=float)

    probs = answer["probabilities"]
    if row["type"] == "choice":
        keys = list(row["criteria"].keys())
        return np.array([float(probs[k]) for k in keys], dtype=float)

    return np.array([float(probs[str(i)]) for i in range(len(row["criteria"]))], dtype=float)


def show_manual_examples(agent):
    rules = (
        "咖啡店規則：所有飲品都是大杯。飲品有美式、拿鐵、燕麥奶拿鐵、鮮奶；"
        "可選冰或熱，也可以加一份濃縮咖啡。"
    )

    examples = [
        "顧客：我要一杯冰拿鐵，加一份濃縮咖啡。",
        "顧客：我想喝咖啡。",
        "顧客：我要一杯藍莓拿鐵。",
    ]

    questions = {
        "next_action": {
            "type": "choice",
            "instructions": "依照目前對話，店員下一步最適合採取哪一種行動？",
            "criteria": {
                "A": "直接確認或整理訂單",
                "B": "詢問缺少或不清楚的資訊",
                "C": "說明目前無法提供，請顧客改選",
            },
        },
        "can_confirm": {
            "type": "noul",
            "instructions": "依照目前對話，資訊是否已足夠讓店員直接確認或整理訂單？",
            "criteria": {
                "false": "資訊還不足，或目前要求無法提供",
                "true": "資訊已足夠，可以直接確認或整理訂單",
            },
        },
        "readiness": {
            "type": "score",
            "instructions": "請評估目前這筆訂單距離可以直接確認還有多遠。",
            "criteria": [
                "目前要求無法提供，需要改選",
                "還缺資訊，需要再問",
                "資訊足夠，可以確認訂單",
            ],
        },
    }

    print("\n========== 三個簡單例子 ==========")
    for text in examples:
        state = rules + "\n\n目前對話：\n" + text
        result = agent.predict(state=state, questions=questions)
        print("\n輸入：", text)
        print(json.dumps(result["answers"], ensure_ascii=False, indent=2))


def benchmark(agent):
    rows = read_jsonl(TEST_FILE)
    if MAX_BENCHMARK_DECISIONS is not None:
        rows = rows[:MAX_BENCHMARK_DECISIONS]

    correct = []
    soft_accuracy = []
    confidence = []
    brier = []
    kl = []
    tv = []
    score_mae = []
    within_one = []
    latencies = []

    for row in rows:
        start = time.perf_counter()
        result = agent.predict(
            state=row["state"],
            questions={"q": external_question(row)},
        )
        latencies.append((time.perf_counter() - start) * 1000)

        answer = result["answers"]["q"]
        p = probabilities_from_result(row, answer)
        target = np.array(row["target"], dtype=float)
        pred = int(np.argmax(p))
        gold = int(np.argmax(target))

        correct.append(float(pred == gold))
        confidence.append(float(np.max(p)))

        # 原 notebook 的 soft accuracy / Brier / KL / TV 主要評估
        # choice 與 noul 的機率分布；score 則另外看 MAE 與 within-one。
        if row["type"] != "score":
            soft_accuracy.append(float(np.sum(p * target)))
            brier.append(float(np.sum((p - target) ** 2)))
            kl.append(float(np.sum(target * np.log(
                np.clip(target, 1e-12, 1.0) / np.clip(p, 1e-12, 1.0)
            ))))
            tv.append(float(0.5 * np.sum(np.abs(p - target))))
        else:
            predicted_score = float(answer["score"])
            score_mae.append(abs(predicted_score - gold))
            within_one.append(float(abs(predicted_score - gold) <= 1.0))

    print("\n========== Held-out classroom benchmark ==========")
    print(f"decisions      : {len(rows)}")
    print(f"accuracy       : {np.mean(correct):.4f}")
    if soft_accuracy:
        print(f"soft accuracy  : {np.mean(soft_accuracy):.4f}")
        print(f"Brier          : {np.mean(brier):.4f}")
        print(f"KL             : {np.mean(kl):.4f}")
        print(f"TV             : {np.mean(tv):.4f}")
    print(f"ECE            : {ece_score(np.array(confidence), np.array(correct)):.4f}")
    if score_mae:
        print(f"score MAE      : {np.mean(score_mae):.4f}")
        print(f"within one     : {np.mean(within_one):.4f}")
    print(f"p50 latency    : {np.median(latencies):.1f} ms")
    print("注意：這是同一公開資料集切出的 held-out 教學測試，不是獨立研究 benchmark。")


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError("找不到微調模型，請先執行 03_train_laya.py。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"載入模型：{MODEL_DIR}")
    agent = laya.load(str(MODEL_DIR), device=device)

    show_manual_examples(agent)
    benchmark(agent)


if __name__ == "__main__":
    main()
