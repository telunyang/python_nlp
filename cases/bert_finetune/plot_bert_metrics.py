"""
plot_bert_metrics.py

讀取 train_bert.py 產生的 trainer_state.json，
輸出：

    output/bert_binary_clf/loss_curve.png

以及如果 log_history 中有 eval_f1：

    output/bert_binary_clf/f1_curve.png

這個版本適合 VS Code + Google Colab Extension 的 remote terminal。

因為 remote terminal 通常沒有桌面 GUI，
所以不依賴 plt.show()，
而是直接使用 Matplotlib Agg backend 把圖存成 PNG。
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =============================================================================
# 1. 路徑設定
# =============================================================================

OUTPUT_DIR = Path("output/bert_binary_clf")

STATE_FILE = OUTPUT_DIR / "trainer_state.json"

LOSS_PLOT_FILE = OUTPUT_DIR / "loss_curve.png"

F1_PLOT_FILE = OUTPUT_DIR / "f1_curve.png"


# =============================================================================
# 2. 主流程
# =============================================================================

def main():

    if not STATE_FILE.is_file():
        raise FileNotFoundError(
            f"找不到：{STATE_FILE.resolve()}\n"
            "請先執行 train_bert.py。"
        )

    with STATE_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        state = json.load(file)

    logs = state.get(
        "log_history",
        [],
    )

    if not logs:
        raise RuntimeError(
            "trainer_state.json 中沒有 log_history。"
        )

    # -------------------------------------------------------------------------
    # Training / Validation Loss
    # -------------------------------------------------------------------------

    train_loss = [
        (item["step"], item["loss"])
        for item in logs
        if "step" in item and "loss" in item
    ]

    valid_loss = [
        (item["step"], item["eval_loss"])
        for item in logs
        if "step" in item and "eval_loss" in item
    ]

    if train_loss or valid_loss:

        plt.figure(figsize=(10, 6))

        if train_loss:
            x, y = zip(*train_loss)
            plt.plot(
                x,
                y,
                label="Training Loss",
                linewidth=1.8,
            )

        if valid_loss:
            x, y = zip(*valid_loss)
            plt.plot(
                x,
                y,
                marker="o",
                label="Validation Loss",
                linewidth=1.8,
            )

        plt.xlabel("Optimizer Step")
        plt.ylabel("Loss")
        plt.title("BERT Training and Validation Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            LOSS_PLOT_FILE,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "Loss 圖表："
            f"{LOSS_PLOT_FILE.resolve()}"
        )

    # -------------------------------------------------------------------------
    # Validation F1
    # -------------------------------------------------------------------------

    eval_f1 = [
        (item["step"], item["eval_f1"])
        for item in logs
        if "step" in item and "eval_f1" in item
    ]

    if eval_f1:

        plt.figure(figsize=(10, 6))

        x, y = zip(*eval_f1)

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            label="Validation F1",
        )

        plt.xlabel("Optimizer Step")
        plt.ylabel("F1")
        plt.title("BERT Validation F1")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            F1_PLOT_FILE,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "F1 圖表："
            f"{F1_PLOT_FILE.resolve()}"
        )

    print(
        f"Training loss points：{len(train_loss)}"
    )
    print(
        f"Validation loss points：{len(valid_loss)}"
    )
    print(
        f"Validation F1 points：{len(eval_f1)}"
    )


if __name__ == "__main__":
    main()
