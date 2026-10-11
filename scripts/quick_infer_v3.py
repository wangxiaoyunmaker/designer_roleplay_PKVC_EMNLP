"""quick_infer_v3.py — 快速试用 stage_warmup LoRA, 3 条 val sample 对照 target.

抽 Q1 / Q2 / Q3-Q6 各 1 条, 用 base + adapter 生成, 跟 ground truth target 对比.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="/data/dhf/chain_critic/model/Qwen2.5-7B-Instruct/")
    p.add_argument("--adapter", default="/data/wy/sft_v3_ckpts/stage_warmup")
    p.add_argument("--val", default="data_v3/processed/sft_val.jsonl")
    p.add_argument("--n-per-q", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.3)
    args = p.parse_args()

    # Lazy imports
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base from {args.base}...")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0",
    )
    print(f"Loading LoRA adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(f"trainable params after load (should be 0 for eval): "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Pick samples
    val_rows = [json.loads(l) for l in open(args.val)]
    by_q = {}
    for r in val_rows:
        by_q.setdefault(r["target_q"], []).append(r)
    picked = []
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"):
        if q in by_q:
            picked.extend(by_q[q][: args.n_per_q])
    print(f"\npicked {len(picked)} samples")

    for sample in picked:
        msgs = [
            {"role": "system", "content": sample["system"]},
            {"role": "user", "content": sample["prompt"]},
        ]
        input_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = tok(input_text, return_tensors="pt", truncation=True, max_length=4096).to("cuda:0")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        gen_ids = out[0, inputs["input_ids"].size(1):]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True).strip()

        print("=" * 70)
        print(f"target_q={sample['target_q']}  designer={sample['user_id']}")
        print("-" * 70)
        print("TARGET (ground truth):")
        print(json.dumps(sample["target"], ensure_ascii=False, indent=2))
        print("-" * 70)
        print("GENERATED:")
        print(gen_text)
        print()


if __name__ == "__main__":
    main()
