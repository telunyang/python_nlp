import json
from collections import Counter
from pathlib import Path

# 這份程式只處理繁體中文，因此不需要翻譯模型，也不需要 OpenCC。
# 如果換成英文或其他語言：
# 1. 01_download_dataset.py 換 DATASET_ID。
# 2. 下面 classify_reply() 的關鍵字規則要換成該語言。
# 3. 三種 instructions / criteria 文字也改成該語言。

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"
DATA_DIR = HERE / "data"

SHOP_RULES = (
    "咖啡店規則：所有飲品都是大杯。飲品有美式、拿鐵、燕麥奶拿鐵、鮮奶；"
    "可選冰或熱，也可以加一份濃縮咖啡。"
)

A_CONFIRM = "A"
B_ASK = "B"
C_REJECT = "C"

ACTION_TEXT = {
    A_CONFIRM: "直接確認或整理訂單",
    B_ASK: "詢問缺少或不清楚的資訊",
    C_REJECT: "說明目前無法提供，請顧客改選",
}


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify_reply(reply):
    """把店員下一句回覆，轉成簡單、可解釋的弱標籤。"""
    text = reply.strip()

    # 明確說商品不存在或不能提供。
    reject_words = ("抱歉", "沒有", "無法提供", "目前只提供", "不提供")
    if any(word in text for word in reject_words):
        return C_REJECT

    # 店員還需要追問資料，例如飲品、冰熱、是否加濃縮。
    if "請問" in text or "？" in text or "?" in text:
        # 已經完整重述訂單，只是最後問「還有其他需求嗎」時，仍視為已確認。
        finished_words = ("以下為您的訂單", "已經為您", "已為您", "為您準備", "為您安排")
        if any(word in text for word in finished_words) and ("其他" in text or "還需要" in text):
            return A_CONFIRM
        return B_ASK

    # 沒有追問或拒絕時，通常就是直接確認、整理訂單。
    return A_CONFIRM


def make_state(history):
    lines = [SHOP_RULES, "", "目前對話："]
    for role, content in history:
        name = "顧客" if role == "user" else "店員"
        lines.append(f"{name}：{content}")
    return "\n".join(lines)


def one_hot(index, size):
    return [1.0 if i == index else 0.0 for i in range(size)]


def convert_conversation(row):
    """一段對話可以產生多個 decision samples。"""
    messages = row.get("conversations", [])
    history = []
    output = []

    # dataset 的 system prompt 很長，而且每筆幾乎相同；我們改用上面的 SHOP_RULES。
    for i, message in enumerate(messages):
        role = message.get("role")
        content = str(message.get("content", "")).strip()

        if role == "system" or not content:
            continue

        if role == "assistant" and history and history[-1][0] == "user":
            action = classify_reply(content)
            state = make_state(history)

            # choice：下一步應採取哪一類行動？
            output.append({
                "state": state,
                "type": "choice",
                "instructions": "依照目前對話，店員下一步最適合採取哪一種行動？",
                "criteria": ACTION_TEXT,
                "target": one_hot([A_CONFIRM, B_ASK, C_REJECT].index(action), 3),
                "label": action,
            })

            # noul：現在是否已經能直接確認訂單？
            can_confirm = action == A_CONFIRM
            output.append({
                "state": state,
                "type": "noul",
                "instructions": "依照目前對話，資訊是否已足夠讓店員直接確認或整理訂單？",
                "criteria": {
                    "false": "資訊還不足，或目前要求無法提供",
                    "true": "資訊已足夠，可以直接確認或整理訂單",
                },
                "target": [0.0, 1.0] if can_confirm else [1.0, 0.0],
                "label": "true" if can_confirm else "false",
            })

            # score：把同一個弱標籤轉成 0/1/2 的「訂單可完成程度」。
            # C -> 0（不能完成）、B -> 1（還要追問）、A -> 2（可確認）。
            score_index = {C_REJECT: 0, B_ASK: 1, A_CONFIRM: 2}[action]
            output.append({
                "state": state,
                "type": "score",
                "instructions": "請評估目前這筆訂單距離可以直接確認還有多遠。",
                "criteria": [
                    "目前要求無法提供，需要改選",
                    "還缺資訊，需要再問",
                    "資訊足夠，可以確認訂單",
                ],
                "target": one_hot(score_index, 3),
                "label": score_index,
            })

        history.append((role, content))

    return output


def convert_split(name):
    rows = read_jsonl(RAW_DIR / f"{name}.jsonl")
    decisions = []
    labels = Counter()

    for row in rows:
        converted = convert_conversation(row)
        decisions.extend(converted)
        for item in converted:
            if item["type"] == "choice":
                labels[item["label"]] += 1

    write_jsonl(decisions, DATA_DIR / f"{name}.jsonl")
    print(f"{name:11s}: {len(rows):4d} 對話 -> {len(decisions):5d} typed decisions")
    if labels:
        print("  choice 標籤：", dict(labels))


def main():
    for split in ("train", "calibration", "test"):
        convert_split(split)

    print("\n轉換完成。每個對話 turn 會產生 choice、noul、score 三種 Laya decision。")
    print("這些 action label 是由店員原始回覆用透明規則產生的 weak labels，不是人工標註。")


if __name__ == "__main__":
    main()
