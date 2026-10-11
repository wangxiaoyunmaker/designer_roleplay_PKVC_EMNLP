"""推理: 方向一 3 步生成 / 方向二 单步生成.

输入 prompt 构造跟 train 时一致 (profile + peer RAG + scenario)
方向一: B1 输出 → 拼回 prompt → B2 → 拼回 → B3
方向二: 单步生成 [PKVC 分析] + [答案]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM

from sft_pipeline.model import load_tokenizer


DELIMITERS = {
    "B1": "[/INTUITION]",
    "B2": "[/WEIGH]",
    "B3": "[/BOTTOM_LINE]",
    "q1q6": "[/答案]",
}


def load_checkpoint(base_name: str, ckpt_dir: Path, dtype=torch.bfloat16):
    base = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=dtype, device_map="auto",
        attn_implementation="sdpa",
    )
    return PeftModel.from_pretrained(base, ckpt_dir).eval()


@torch.no_grad()
def generate_segment(model, tokenizer, prompt_text: str, stop_marker: str,
                     max_new_tokens: int = 512) -> str:
    # Match training tokenization (dataset.py:47): add trailing \n, no special tokens
    ids = tokenizer(prompt_text + "\n", return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0, ids.size(1):], skip_special_tokens=True)
    # 在 stop_marker 处截断
    if stop_marker in text:
        text = text[:text.index(stop_marker) + len(stop_marker)]
    return text


def infer_q7_direction(model, tokenizer, prompt_base: str, max_new_tokens: int = 512) -> dict:
    """方向一: 3 步生成 B1 → B2 → B3."""
    cur_prompt = prompt_base
    seg_b1 = generate_segment(model, tokenizer, cur_prompt, DELIMITERS["B1"], max_new_tokens)
    cur_prompt = cur_prompt + "\n" + seg_b1
    seg_b2 = generate_segment(model, tokenizer, cur_prompt, DELIMITERS["B2"], max_new_tokens)
    cur_prompt = cur_prompt + "\n" + seg_b2
    seg_b3 = generate_segment(model, tokenizer, cur_prompt, DELIMITERS["B3"], max_new_tokens)
    return {"B1": seg_b1, "B2": seg_b2, "B3": seg_b3,
             "full_response": "\n".join([seg_b1, seg_b2, seg_b3])}


def infer_q1q6_direction(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> dict:
    """方向二: 单步生成 [PKVC 分析] + [答案]."""
    full = generate_segment(model, tokenizer, prompt, DELIMITERS["q1q6"], max_new_tokens)
    return {"full_response": full}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sft_pipeline/config/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-jsonl", required=True, help="测试 jsonl, 每行含 prompt")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--direction", choices=["q7", "q1q6"], required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tokenizer = load_tokenizer(cfg["base_model"]["name"])
    model = load_checkpoint(cfg["base_model"]["name"], Path(args.checkpoint))

    inputs = [json.loads(l) for l in open(args.input_jsonl, encoding="utf-8")]
    results = []
    infer_fn = infer_q7_direction if args.direction == "q7" else infer_q1q6_direction
    for i, item in enumerate(inputs):
        result = infer_fn(model, tokenizer, item["input_text"],
                          max_new_tokens=cfg["inference"]["max_new_tokens"])
        results.append({"user_id": item["user_id"], "task_id": item.get("task_id"),
                         "prediction": result, "meta": item.get("meta", {})})
        if i % 10 == 0:
            print(f"  done {i}/{len(inputs)}")

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} predictions to: {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
