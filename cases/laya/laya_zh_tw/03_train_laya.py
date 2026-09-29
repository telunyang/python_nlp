import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

import laya
from laya.common import QTYPES, build_sequence, collate_items, proper_reward, render_options

# ============================================================
# 單張 Tesla T4 的 Laya RLCD 教學版
# ============================================================
# 這裡故意不用 CONFIG dict。學生要改設定，直接改下面幾個值。
#
# 繁體中文 / 多國語系：convaiinnovations/laya-multilingual
# 純英文：把 BASE_MODEL 改成 convaiinnovations/laya
BASE_MODEL = "convaiinnovations/laya-multilingual"

# 跑幾輪資料。越大通常學得更多，但也更慢，且可能過度記住訓練資料。
EPOCHS = 2

# 一次放進 GPU 的 decision 數量。T4 建議從 2 開始；顯存夠可試 4。
MICRO_BATCH = 2

# 累積幾次小 batch 才更新一次。MICRO_BATCH=2、GRAD_ACCUM=32 -> effective batch 約 64。
GRAD_ACCUM = 32

# 每個答案旁邊製造幾個探索版本。原 notebook 的核心設定是 4。
GROUP_SIZE = 4

# Encoder 改慢一點，Decision head 改快一點。
LR_ENCODER = 2.5e-5
LR_HEAD = 1.0e-4
WEIGHT_DECAY = 0.01

# sigma 是探索幅度：前期多嘗試，後期逐漸穩定。
SIGMA_START = 0.4
SIGMA_END = 0.1

# 保留 RLCD proper scoring reward 與 CE guidance。
W_SPH = 0.75
W_RPS = 1.0
CE_WEIGHT = 1.0

# 對話通常不長；1024 可保留足夠上下文。
MAX_LEN = 1024
HEAD_MAX_LEN = 256
MAX_GRAD_NORM = 1.0

# 校準不用全部資料，600 個 typed decisions 已足夠做教學示範。
CALIBRATION_LIMIT = 600
CALIBRATION_BATCH_SIZE = 16
SEED = 42

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "output" / "laya_coffee_zh_tw"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def internal_question(row):
    return {
        "t": row["type"],
        "ins": row["instructions"],
        "crit": row.get("criteria"),
    }


def build_item(row, tokenizer):
    q = internal_question(row)
    ids, markers = build_sequence(
        tokenizer,
        row["state"],
        q,
        max_len=MAX_LEN,
        head_max_len=HEAD_MAX_LEN,
    )

    expected_options = len(render_options(q))
    if len(markers) != expected_options or len(row["target"]) != expected_options:
        return None

    return {
        "ids": ids,
        "markers": markers,
        "qtype": QTYPES[row["type"]],
        "target": row["target"],
        "label": int(np.argmax(row["target"])),
    }


def build_items(path, tokenizer):
    items = []
    skipped = 0
    for row in read_jsonl(path):
        item = build_item(row, tokenizer)
        if item is None:
            skipped += 1
        else:
            items.append(item)
    return items, skipped


def move_batch(batch, device):
    for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "target"):
        batch[key] = batch[key].to(device)
    return batch


def amp_context():
    return torch.autocast("cuda", dtype=torch.float16) if torch.cuda.is_available() else nullcontext()


def fit_temperature(samples):
    """用 held-out calibration logits 找一個 temperature。"""
    if len(samples) < 10:
        return 1.0

    kmax = max(len(logits) for logits, _ in samples)
    logits = torch.full((len(samples), kmax), -1e4, dtype=torch.float32)
    targets = torch.zeros((len(samples), kmax), dtype=torch.float32)

    for i, (z, target) in enumerate(samples):
        logits[i, : len(z)] = torch.tensor(z, dtype=torch.float32)
        targets[i, : len(target)] = torch.tensor(target, dtype=torch.float32)

    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = -(targets * torch.log_softmax(logits / log_t.exp(), dim=-1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(log_t.exp(), 0.5, 5.0).item())


def save_model(model, tokenizer, cfg):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    state = {}
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu().contiguous()
        state[name] = tensor.half() if torch.is_floating_point(tensor) else tensor

    save_file(state, str(OUTPUT_DIR / "model.safetensors"))
    model.encoder.config.save_pretrained(OUTPUT_DIR / "encoder")
    tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer")

    with (OUTPUT_DIR / "rl_agent_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("請在 Colab 選擇 NVIDIA GPU，例如 Tesla T4。")
    if GROUP_SIZE < 2:
        raise ValueError("GROUP_SIZE 至少要 2，否則 group baseline 沒有比較對象。")
    if SIGMA_START <= 0 or SIGMA_END <= 0:
        raise ValueError("SIGMA_START 與 SIGMA_END 都必須大於 0。")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda")

    print(f"載入 base model：{BASE_MODEL}")
    # 直接使用 laya.load()，省掉手動下載 config / tokenizer / safetensors 的程式。
    base_agent = laya.load(BASE_MODEL, device="cpu")
    tokenizer = base_agent.tok
    model = base_agent.model
    cfg = dict(base_agent.cfg)
    del base_agent

    cfg["max_len"] = MAX_LEN
    cfg["head_max_len"] = HEAD_MAX_LEN
    cfg["gradient_checkpointing"] = True

    # Gradient checkpointing 用更多計算換更少顯存，單張 T4 很重要。
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        try:
            model.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.encoder.gradient_checkpointing_enable()
    if hasattr(model.encoder.config, "use_cache"):
        model.encoder.config.use_cache = False
    model.head_checkpointing = True
    model.to(device).train()

    train_items, skipped_train = build_items(DATA_DIR / "train.jsonl", tokenizer)
    calibration_items, skipped_cal = build_items(DATA_DIR / "calibration.jsonl", tokenizer)
    calibration_items = calibration_items[:CALIBRATION_LIMIT]

    if not train_items or not calibration_items:
        raise RuntimeError("沒有可用資料。請先執行 01_download_dataset.py 與 02_convert_dataset.py。")

    encoder_params = list(model.encoder.parameters())
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": LR_ENCODER},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    micro_batches_per_epoch = math.ceil(len(train_items) / MICRO_BATCH)
    updates_per_epoch = math.ceil(micro_batches_per_epoch / GRAD_ACCUM)
    total_updates = max(1, EPOCHS * updates_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda")

    print("\n========== 訓練設定 ==========")
    print(f"train decisions       : {len(train_items)} (skipped={skipped_train})")
    print(f"calibration decisions : {len(calibration_items)} (skipped={skipped_cal})")
    print(f"micro batch           : {MICRO_BATCH}")
    print(f"gradient accumulation : {GRAD_ACCUM}")
    print(f"effective batch       : {MICRO_BATCH * GRAD_ACCUM}")
    print(f"group size            : {GROUP_SIZE}")
    print(f"epochs                : {EPOCHS}")
    print("==============================\n")

    start_time = time.time()
    global_update = 0

    for epoch in range(EPOCHS):
        random.Random(SEED + epoch).shuffle(train_items)
        micro_batches = [
            train_items[i : i + MICRO_BATCH]
            for i in range(0, len(train_items), MICRO_BATCH)
        ]

        # sigma 從大慢慢變小。
        progress = epoch / max(1, EPOCHS - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress
        epoch_loss = 0.0
        epoch_reward = 0.0
        seen_micro_batches = 0

        for group_start in range(0, len(micro_batches), GRAD_ACCUM):
            accumulation_group = micro_batches[group_start : group_start + GRAD_ACCUM]
            optimizer.zero_grad(set_to_none=True)

            for chunk in accumulation_group:
                batch = collate_items([chunk], tokenizer.pad_token_id)
                batch = move_batch(batch, device)

                with amp_context():
                    logits, _ = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["marker_pos"],
                        batch["marker_mask"],
                        batch["qtype"],
                    )

                # RLCD 計算改用 float32，數值比較穩定。
                logits = logits.float()
                mask = batch["marker_mask"]
                target = batch["target"]
                qtype = batch["qtype"]
                option_count = mask.sum(-1, keepdim=True).float().clamp_min(1.0)

                # 1. Gaussian exploration
                eps = torch.randn(
                    (GROUP_SIZE,) + tuple(logits.shape), device=device
                ) * sigma * mask
                eps = (eps - eps.sum(-1, keepdim=True) / option_count) * mask
                sampled_logits = logits.detach().unsqueeze(0) + eps
                q = torch.softmax(sampled_logits.masked_fill(~mask, -1e4), dim=-1)

                # 2. Proper scoring reward + GRPO-style group baseline
                with torch.no_grad():
                    reward = proper_reward(
                        q,
                        target.unsqueeze(0),
                        qtype,
                        mask,
                        w_sph=W_SPH,
                        w_rps=W_RPS,
                    )
                    advantage = reward - reward.mean(0, keepdim=True)
                    advantage = advantage / (advantage.std(unbiased=False) + 1e-6)

                # 3. Policy-gradient loss
                logp = -(
                    ((sampled_logits - logits.unsqueeze(0)) ** 2) * mask
                ).sum(-1) / (2 * sigma**2)
                loss_rl = -(advantage * logp).mean()

                # 4. Soft cross-entropy guidance
                loss_ce = -(
                    target
                    * torch.log_softmax(logits.masked_fill(~mask, -1e4), dim=-1)
                ).sum(-1).mean()

                full_loss = loss_rl + CE_WEIGHT * loss_ce
                loss = full_loss / len(accumulation_group)
                scaler.scale(loss).backward()

                epoch_loss += float(full_loss.detach())
                epoch_reward += float(reward.mean())
                seen_micro_batches += 1

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_update += 1

            if global_update % 20 == 0:
                print(
                    f"epoch={epoch + 1}/{EPOCHS} update={global_update} "
                    f"loss={epoch_loss / seen_micro_batches:.4f} "
                    f"reward={epoch_reward / seen_micro_batches:.3f} "
                    f"sigma={sigma:.3f}"
                )

        print(
            f"Epoch {epoch + 1} 完成，平均 loss={epoch_loss / max(1, seen_micro_batches):.4f}，"
            f"平均 reward={epoch_reward / max(1, seen_micro_batches):.3f}"
        )

    # ========================================================
    # Held-out temperature calibration
    # ========================================================
    print("\n開始 temperature calibration...")
    model.eval()
    by_type = {0: [], 1: [], 2: []}

    with torch.inference_mode():
        for start in range(0, len(calibration_items), CALIBRATION_BATCH_SIZE):
            chunk = calibration_items[start : start + CALIBRATION_BATCH_SIZE]
            batch = move_batch(collate_items([chunk], tokenizer.pad_token_id), device)
            with amp_context():
                logits, _ = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["marker_pos"],
                    batch["marker_mask"],
                    batch["qtype"],
                )

            logits = logits.float().cpu().numpy()
            for i, item in enumerate(chunk):
                k = len(item["markers"])
                by_type[item["qtype"]].append((logits[i, :k], item["target"]))

    temperatures = [fit_temperature(by_type[i]) for i in range(3)]
    print("temperature [choice, score, noul] =", [round(x, 3) for x in temperatures])

    # Laya 載入本地模型需要原本的 rl_agent_config.json。
    # 這裡只更新真的會影響推論的 temperature，不另外建立教學用設定 dict。
    cfg["temperature"] = temperatures
    cfg.pop("temperature_by_options", None)

    save_model(model, tokenizer, cfg)
    print(f"\n完成。模型存到：{OUTPUT_DIR}")
    print(f"總時間：{(time.time() - start_time) / 60:.1f} 分鐘")
    print("下一步執行：python 04_test_laya.py")


if __name__ == "__main__":
    main()
