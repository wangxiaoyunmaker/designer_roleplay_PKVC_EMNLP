"""q2_first_letter_infer.py — 评估 Q2 fix model 是否学到 minority first-letter.

对每个 first-letter class (P/C/K/V) 各选 ~3 designer, 跑 Q2 inference, 看
generated thinking_chain 第 1 个 cardId 是否跟 designer-specific target 一致.
"""
import argparse
import json
import sys
from pathlib import Path

# Target designer per first letter (picked from raw Q2 labels analysis)
PICKS = {
    "C": ["vivienfangshiwan", "-DevolS", "1142208878"],
    "K": ["C-0223-C-", "13893214526", "F29293918"],
    "V": ["17317216377", "GalaxieFiveHundred", "Ellieqiuchi1003"],
    "P": ["15605160898", "13224246380", "15102954680"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="/data/dhf/chain_critic/model/Qwen2.5-7B-Instruct/")
    p.add_argument("--adapter", default="/data/wy/sft_v3_ckpts/stage_warmup_ep5_q2fix")
    p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.3)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base + adapter {args.adapter}...")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # Load Q2 from sft_train + sft_val
    by_uid = {}
    for path in ("data_v3/processed/sft_train.jsonl", "data_v3/processed/sft_val.jsonl"):
        for line in open(path):
            r = json.loads(line)
            if r["target_q"] == "Q2":
                by_uid[r["user_id"]] = r

    print(f"\n{'='*80}")
    print(f"Q2 first-letter accuracy by class")
    print(f"{'='*80}")

    summary = {"P": [0, 0], "K": [0, 0], "V": [0, 0], "C": [0, 0]}  # [hits, total]

    for letter, uids in PICKS.items():
        print(f"\n--- {letter}-first targets ---")
        for uid in uids:
            sample = by_uid.get(uid)
            if not sample:
                print(f"  {uid}: NOT IN sft data (skipped)")
                continue
            msgs = [
                {"role": "system", "content": sample["system"]},
                {"role": "user", "content": sample["prompt"]},
            ]
            input_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok(input_text, return_tensors="pt", truncation=True, max_length=4096).to("cuda:0")
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature > 0), temperature=args.temperature,
                    top_p=0.9, pad_token_id=tok.eos_token_id,
                )
            gen = tok.decode(out[0, inputs["input_ids"].size(1):], skip_special_tokens=True).strip()
            # Parse JSON
            try:
                obj = json.loads(gen)
                gen_chain = obj.get("thinking_chain", [])
                gen_first = gen_chain[0] if gen_chain else None
                gen_first_letter = gen_first[0] if gen_first else None
            except Exception as e:
                gen_chain, gen_first, gen_first_letter = [], None, f"PARSE_ERR ({e})"
            tgt_chain = sample["target"]["thinking_chain"]
            tgt_first = tgt_chain[0]
            hit = (gen_first_letter == letter)
            summary[letter][1] += 1
            if hit:
                summary[letter][0] += 1
            mark = "✓" if hit else "✗"
            print(f"  {mark} {uid}: target={tgt_first} (chain {tgt_chain[:5]}...), gen={gen_first} (chain {gen_chain[:5]}...) [letter {'match' if hit else 'MISS'}]")

    print(f"\n{'='*80}")
    print("Summary: Q2 first-letter accuracy")
    for L in "PCKV":
        h, t = summary[L]
        print(f"  {L}-first: {h}/{t}{' ('+str(100*h//t)+'%)' if t else ''}")
    total_h = sum(s[0] for s in summary.values())
    total_t = sum(s[1] for s in summary.values())
    print(f"  Overall: {total_h}/{total_t} ({100*total_h//total_t if total_t else 0}%)")


if __name__ == "__main__":
    main()
