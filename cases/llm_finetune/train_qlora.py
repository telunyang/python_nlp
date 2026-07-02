# train_qlora.py
'''
執行 non-thinking：
python train_qlora.py \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --data_dir data/sft_no_thinking \
  --output_dir outputs/qwen25_05b_no_thinking \
  --max_length 768 \
  --grad_accum 16

執行 thinking，建議先用 Qwen3：
python train_qlora.py \
  --model_id Qwen/Qwen3-0.6B \
  --data_dir data/sft_thinking \
  --output_dir outputs/qwen3_06b_thinking \
  --max_length 1024 \
  --grad_accum 16

  
如果爆 VRAM，依序調小：
max_length: 1024 → 768 → 512
lora r: 8 → 4
grad_accum: 保留 16 或 32，不會直接增加單步 VRAM
per_device_train_batch_size: 固定 1


檢視收斂過程與 early stopping
tensorboard --logdir outputs/qwen25_05b_no_thinking/runs
'''
import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_dir", default="data/sft_no_thinking")
    parser.add_argument("--output_dir", default="outputs/qwen_lora")
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--eval_steps", type=int, default=50)
    return parser.parse_args()


def load_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def build_lora_config():
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = get_args()

    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{args.data_dir}/train.jsonl",
            "validation": f"{args.data_dir}/valid.jsonl",
        },
    )

    model, tokenizer = load_model_and_tokenizer(args.model_id)

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.lr,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        optim="paged_adamw_8bit",

        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        logging_strategy="steps",
        logging_steps=10,
        logging_dir=f"{args.output_dir}/runs",
        report_to=["tensorboard"],

        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        completion_only_loss=True,
        packing=False,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if len(dataset["validation"]) else None,
        processing_class=tokenizer,
        peft_config=build_lora_config(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()