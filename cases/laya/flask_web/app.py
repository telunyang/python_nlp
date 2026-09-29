from pathlib import Path

import torch
from flask import Flask, render_template, request
import laya

# ===== 最常需要改的設定 =====
# 如果你換了模型輸出資料夾，只要改 MODEL_DIR。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "laya_zh_tw" / "output" / "laya_coffee_zh_tw"
HOST = "0.0.0.0"
PORT = 5000

SHOP_RULES = (
    "咖啡店規則：所有飲品都是大杯。飲品有美式、拿鐵、燕麥奶拿鐵、鮮奶；"
    "可選冰或熱，也可以加一份濃縮咖啡。"
)

app = Flask(__name__)
_laya = None


def questions():
    """Laya 需要的三種 typed decisions。"""
    return {
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


def get_laya():
    """第一次收到請求時才載入模型，之後重複使用同一份模型。"""
    global _laya

    if _laya is None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                "找不到微調模型。請先執行 laya_zh_tw/03_train_laya.py。"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"載入 Laya：{MODEL_DIR}")
        print(f"使用裝置：{device}")
        _laya = laya.load(str(MODEL_DIR), device=device)

    return _laya


def predict_order(user_text):
    """把使用者輸入交給 Laya，一次取得 choice、noul、score。"""
    state = SHOP_RULES + "\n\n目前對話：\n顧客：" + user_text
    return get_laya().predict(state=state, questions=questions())["answers"]


@app.route("/", methods=["GET", "POST"])
def index():
    user_text = ""
    answers = None
    error = None

    if request.method == "POST":
        user_text = request.form.get("user_text", "").strip()

        if not user_text:
            error = "請先輸入一段咖啡點餐內容。"
        else:
            try:
                answers = predict_order(user_text)
            except Exception as exc:
                # 教學展示時直接把錯誤顯示在網頁上，方便學生看懂問題。
                error = str(exc)

    return render_template(
        "index.html",
        user_text=user_text,
        answers=answers,
        error=error,
        questions=questions(),
    )


if __name__ == "__main__":
    # 這是課堂與本機展示用途。
    # 正式公開服務不要直接使用 Flask development server。
    app.run(host=HOST, port=PORT, debug=False)
