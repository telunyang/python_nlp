# prepare_chat_data.py
'''
python prepare_chat_data.py \
  --history chat_history.json \
  --out_dir data/sft_no_thinking \
  --mode no_thinking

python prepare_chat_data.py \
  --history chat_history.json \
  --out_dir data/sft_thinking \
  --mode thinking
'''
import argparse
import json
import random
import re
from pathlib import Path

ROLE_MAP = {
    "user": "user",
    "model": "assistant",
    "assistant": "assistant",
    "system": "system",
}

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def load_history(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    messages = []
    for m in raw:
        role = ROLE_MAP.get(m.get("role"))
        text = (m.get("text") or m.get("content") or "").strip()

        if role and text:
            messages.append({"role": role, "content": text})

    return messages


def strip_thinking(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def build_examples(
    messages: list[dict],
    system_prompt: str,
    max_context_messages: int,
    mode: str,
) -> list[dict]:
    examples = []

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        context = messages[:i]
        if not context or context[-1]["role"] != "user":
            continue

        prompt = [{"role": "system", "content": system_prompt}]
        prompt.extend(context[-max_context_messages:])

        answer = msg["content"].strip()
        if mode == "no_thinking":
            answer = strip_thinking(answer)

        if not answer:
            continue

        examples.append({
            "prompt": prompt,
            "completion": [{"role": "assistant", "content": answer}],
        })

    return examples


def save_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="chat_history.json")
    parser.add_argument("--out_dir", default="data/sft")
    parser.add_argument("--mode", choices=["no_thinking", "thinking"], default="no_thinking")
    parser.add_argument("--max_context_messages", type=int, default=8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    system_prompt = (
        "你是一個在地端運行的繁體中文多輪對話助理。"
        "回答要清楚、簡潔、可靠；不知道時要明確說不知道。"
    )

    messages = load_history(args.history)
    examples = build_examples(
        messages=messages,
        system_prompt=system_prompt,
        max_context_messages=args.max_context_messages,
        mode=args.mode,
    )

    random.seed(args.seed)
    random.shuffle(examples)

    n_valid = max(1, int(len(examples) * args.valid_ratio)) if len(examples) > 10 else 0
    valid = examples[:n_valid]
    train = examples[n_valid:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(train, out_dir / "train.jsonl")
    save_jsonl(valid, out_dir / "valid.jsonl")

    print(f"train={len(train)}, valid={len(valid)}")


if __name__ == "__main__":
    main()