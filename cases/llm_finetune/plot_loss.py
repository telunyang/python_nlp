# plot_loss.py
import json
import pandas as pd
import matplotlib.pyplot as plt

state_path = "outputs/qwen25_05b_no_thinking/checkpoint-50/trainer_state.json"

with open(state_path, "r", encoding="utf-8") as f:
    logs = json.load(f)["log_history"]

df = pd.DataFrame(logs)

if "loss" in df:
    df.dropna(subset=["loss"]).plot(x="step", y="loss", title="train loss")
    plt.show()

if "eval_loss" in df:
    df.dropna(subset=["eval_loss"]).plot(x="step", y="eval_loss", title="eval loss")
    plt.show()