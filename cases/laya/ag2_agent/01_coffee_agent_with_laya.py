import asyncio
import json
import os
from pathlib import Path

from ag2 import Agent, tool
from ag2.config import OllamaConfig
import laya

# 如果 Ollama 在別台機器或 Colab + ngrok，只要設定 OLLAMA_BASE_URL。
OLLAMA_MODEL = "qwen3.5:4b"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "laya_zh_tw" / "output" / "laya_coffee_zh_tw"
_LAYA = None

SHOP_RULES = (
    "咖啡店規則：所有飲品都是大杯。飲品有美式、拿鐵、燕麥奶拿鐵、鮮奶；"
    "可選冰或熱，也可以加一份濃縮咖啡。"
)


def get_laya():
    global _LAYA
    if _LAYA is None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError("找不到微調模型，請先執行 laya_zh_tw/03_train_laya.py。")
        _LAYA = laya.load(str(MODEL_DIR), device="cuda")
    return _LAYA


@tool
def consult_laya_order(user_text: str) -> str:
    """請 Laya 判斷咖啡店現在應該確認訂單、繼續追問，或請顧客改選。"""
    state = SHOP_RULES + "\n\n目前對話：\n顧客：" + user_text
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
    result = get_laya().predict(state=state, questions=questions)
    return json.dumps(result["answers"], ensure_ascii=False)


async def main():
    # 如果改成英文或其他語言：
    # 1. Laya 訓練資料與 prompts 要換成相同語言。
    # 2. 純英文 Laya 可改用 convaiinnovations/laya 作為 base model。
    ollama = OllamaConfig(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        streaming=False,
    )

    agent = Agent(
        "CoffeeAgent",
        prompt=(
            "你是一位臺灣咖啡店員，只使用繁體中文。"
            "收到顧客點餐後，先呼叫 consult_laya_order。"
            "依照 Laya 的 next_action、can_confirm 與 readiness 決定要直接確認、追問資訊，"
            "或告知品項無法提供。回答保持簡短自然。"
        ),
        config=ollama,
        tools=[consult_laya_order],
    )

    user_text = input("請輸入咖啡訂單：\n> ").strip()
    if not user_text:
        user_text = "我要一杯冰拿鐵，加一份濃縮咖啡。"

    reply = await agent.ask(user_text)
    print("\nAG2 店員：")
    print(await reply.content())


if __name__ == "__main__":
    asyncio.run(main())
