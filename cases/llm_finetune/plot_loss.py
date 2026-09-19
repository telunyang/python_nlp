"""
讀取 train_qlora.py 最後儲存的 trainer_state.json，
把 training loss 與 validation loss 畫在同一張圖。

這個版本特別適合：
- VS Code + Google Colab Extension
- 在 Colab Terminal 中執行 Python 腳本
- 沒有桌面 GUI 的遠端 Linux 環境

重點：
plt.show() 在沒有圖形介面的 terminal 環境中通常不會跳出視窗，
因此這裡改成直接輸出 PNG 檔案。
"""

from pathlib import Path
import json

# 必須在 import matplotlib.pyplot 之前指定非 GUI backend。
# "Agg" 會把圖畫到檔案，而不是嘗試開啟桌面視窗。
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =============================================================================
# 設定
# =============================================================================

# train_qlora.py 的輸出目錄。
OUTPUT_DIR = Path("outputs/qwen25_05b_lora")

# Trainer state 檔案。
STATE_FILE = OUTPUT_DIR / "trainer_state.json"

# 圖表輸出位置。
PLOT_FILE = OUTPUT_DIR / "loss_curve.png"


def main() -> None:
    """
    讀取 Hugging Face Trainer 的 log_history，
    畫出 training loss 與 validation loss，
    並將圖片存成 PNG。
    """

    # -------------------------------------------------------------------------
    # Step 1：確認 trainer_state.json 是否存在
    # -------------------------------------------------------------------------

    if not STATE_FILE.is_file():
        raise FileNotFoundError(
            f"找不到 Trainer state：{STATE_FILE}\n"
            "請確認 train_qlora.py 已正常執行 trainer.save_state()，"
            "或檢查 OUTPUT_DIR 是否正確。"
        )

    # -------------------------------------------------------------------------
    # Step 2：讀取 Trainer state
    # -------------------------------------------------------------------------

    with STATE_FILE.open("r", encoding="utf-8") as file:
        state = json.load(file)

    logs = state.get("log_history", [])

    if not logs:
        raise RuntimeError(
            f"{STATE_FILE} 裡沒有 log_history，無法繪製 loss curve。"
        )

    # -------------------------------------------------------------------------
    # Step 3：取出 training loss
    # -------------------------------------------------------------------------

    # Hugging Face Trainer 的 training log 通常長這樣：
    #
    # {
    #     "loss": 1.234,
    #     "learning_rate": 0.0001,
    #     "epoch": 0.5,
    #     "step": 10
    # }
    #
    # 我們只保留同時具有 step 與 loss 的項目。
    train = [
        (item["step"], item["loss"])
        for item in logs
        if "step" in item and "loss" in item
    ]

    # -------------------------------------------------------------------------
    # Step 4：取出 validation loss
    # -------------------------------------------------------------------------

    # Evaluation log 通常長這樣：
    #
    # {
    #     "eval_loss": 1.102,
    #     "eval_runtime": ...,
    #     "epoch": ...,
    #     "step": 50
    # }
    valid = [
        (item["step"], item["eval_loss"])
        for item in logs
        if "step" in item and "eval_loss" in item
    ]

    # -------------------------------------------------------------------------
    # Step 5：確認至少有一種 loss
    # -------------------------------------------------------------------------

    if not train and not valid:
        available_keys = sorted(
            {
                key
                for item in logs
                for key in item.keys()
            }
        )

        raise RuntimeError(
            "log_history 中找不到 loss 或 eval_loss。\n"
            f"目前可看到的欄位有：{available_keys}"
        )

    print(f"Training loss points：{len(train)}")
    print(f"Validation loss points：{len(valid)}")

    # -------------------------------------------------------------------------
    # Step 6：建立圖表
    # -------------------------------------------------------------------------

    # figsize 單位是 inch。
    # 10 × 6 對一般螢幕與報告都很好閱讀。
    plt.figure(figsize=(10, 6))

    if train:
        train_steps, train_losses = zip(*train)

        plt.plot(
            train_steps,
            train_losses,
            label="Training Loss",
            linewidth=1.8,
        )

    if valid:
        valid_steps, valid_losses = zip(*valid)

        plt.plot(
            valid_steps,
            valid_losses,
            marker="o",
            markersize=5,
            label="Validation Loss",
            linewidth=1.8,
        )

    # -------------------------------------------------------------------------
    # Step 7：設定圖表文字
    # -------------------------------------------------------------------------

    plt.xlabel("Optimizer Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid(alpha=0.3)

    # 自動調整邊界，避免標題或座標文字被切掉。
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Step 8：儲存圖片
    # -------------------------------------------------------------------------

    # dpi=150：
    # 對一般螢幕與報告足夠清楚。
    #
    # bbox_inches="tight"：
    # 移除不必要的外部空白。
    plt.savefig(
        PLOT_FILE,
        dpi=150,
        bbox_inches="tight",
    )

    # 關閉 figure，避免程式重複執行時累積記憶體。
    plt.close()

    print(f"Loss 圖表已輸出：{PLOT_FILE.resolve()}")


if __name__ == "__main__":
    main()
