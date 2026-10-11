"""run_sft_inference.py — SFT (Qwen2.5-7B + LoRA) inference for B6 prompt variants on Q7 seg1/2/3.

复用 scripts/run_b6_sample.py + scripts/run_3_baselines_q7.py 的 prompt builder + data loader,
替换 OpenAI backend 为本地 Qwen + LoRA. JSON schema 嵌 prompt (Qwen 无 strict schema).

CLI 镜像 run_b6_sample.py:
    python scripts/run_sft_inference.py \\
        --variant v2 --n 0 --segs seg1,seg2,seg3 \\
        --adapter /data/wy/sft_v3_ckpts/stage_warmup_ep5 \\
        --output reports/datav3/sft_v2_inference.jsonl

输出 jsonl 跟 run_b6_sample 同 schema, 兼容 scripts/eval_baselines_pkvc.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Make sibling scripts importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_b6_sample import (  # noqa: E402
    SAMPLE_DESIGNERS,
    build_b6_prompt,
    format_caps_quad_balanced_nl,
    format_self_q3q6_b6,
    get_activated_quads_balanced,
    get_caps_strategy_for_seg,
    get_system_prompt_for_seg,
    load_pi_5,
    polarity_to_stance_block,
)
from scripts.run_3_baselines_q7 import (  # noqa: E402
    _format_peer_q7_block,
    build_seg_schema,
    format_activated_quads_nl,
    get_activated_quads,
    load_caps_pool,
    load_caps_quadruples,
    load_designer_scores,
    load_self_q1q6,
    retrieve_caps_peer_q7_nl,
)


# ────────────────────────────────────────────────────────────────────
# JSON parsing (robust, Qwen 不严格 schema)
# ────────────────────────────────────────────────────────────────────

_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def parse_json_robust(text: str) -> tuple[dict | None, str]:
    """Return (parsed_dict_or_None, error_msg). 尝试 strip markdown / 找第一个 {...} 块."""
    txt = (text or "").strip()
    if not txt:
        return None, "empty"
    # Strip markdown fence
    m = _FENCE.search(txt)
    if m:
        txt = m.group(1).strip()
    # Try direct parse
    try:
        return json.loads(txt), ""
    except json.JSONDecodeError:
        pass
    # Find first {...} balanced block
    start = txt.find("{")
    if start < 0:
        return None, "no_brace"
    depth = 0
    for i in range(start, len(txt)):
        c = txt[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(txt[start:i + 1]), ""
                except json.JSONDecodeError as e:
                    return None, f"balanced_parse_err: {e}"
    return None, "unbalanced_braces"


# ────────────────────────────────────────────────────────────────────
# Prompt schema injection (Qwen 无 strict schema, 嵌入 prompt)
# ────────────────────────────────────────────────────────────────────

def augment_system_with_schema(system_prompt: str, schema: dict) -> str:
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
    return (
        system_prompt
        + "\n\n## 严格按以下 JSON Schema 输出 (必须返回有效 JSON, 所有 required 字段都要有):\n"
        + f"```json\n{schema_str}\n```\n"
        + "只返回 JSON 对象, 不要任何额外文字 / markdown 围栏 / 注释."
    )


# ────────────────────────────────────────────────────────────────────
# Build jobs (跟 run_b6_sample.main 同逻辑, 抽出来)
# ────────────────────────────────────────────────────────────────────

def build_jobs(args, pool, pi, scores, self_q1q6, quads, target_designers):
    modules = {m.strip() for m in args.modules.split(",") if m.strip()}
    jobs = []
    for did in target_designers:
        if did not in pi:
            continue
        pi_block = pi[did]
        stance_block = polarity_to_stance_block(scores.get(did, {}))
        self_block = format_self_q3q6_b6(self_q1q6.get(did, {}))
        activated_balanced = get_activated_quads_balanced(scores.get(did, {}), quads)
        if args.variant == "v2_no_beta_summary":
            caps_balanced = format_caps_quad_balanced_nl(activated_balanced, variant="v2_no_beta_summary")
        else:
            caps_balanced = format_caps_quad_balanced_nl(activated_balanced, variant="v2")
        activated_topk = get_activated_quads(scores.get(did, {}), quads, top_k=5)
        caps_topk = format_activated_quads_nl(activated_topk)

        target_segs = [s.strip() for s in args.segs.split(",") if s.strip()]
        for seg in target_segs:
            scenario = f"Q7_{seg}"
            if did not in pool.designers or not pool.designers[did].has_scenario(scenario):
                continue
            caps_strategy = get_caps_strategy_for_seg(args.variant, seg)
            caps_block = caps_balanced if caps_strategy == "balanced" else caps_topk
            effective_variant = "v2_clean" if caps_strategy == "topk" else "v2"
            peer_block = ""
            if args.with_peer:
                peers = retrieve_caps_peer_q7_nl(did, pool, scenario, k_peers=3)
                peer_block = _format_peer_q7_block(peers, "认知模式相似度")
            prompt = build_b6_prompt(
                scenario, pi_block, stance_block, self_block, caps_block,
                variant=effective_variant, peer_block=peer_block,
                modules=modules,
            )
            system_prompt = get_system_prompt_for_seg(args.variant, seg)
            jobs.append({
                "designer_id": did,
                "scenario": scenario,
                "seg": seg,
                "prompt": prompt,
                "system_prompt": system_prompt,
            })
    return jobs


# ────────────────────────────────────────────────────────────────────
# Local Qwen + LoRA inference
# ────────────────────────────────────────────────────────────────────

def load_qwen_lora(base_path: str, adapter_path: str | None, dtype="bf16", quantize="none"):
    """adapter_path = "" / "none" / None → base only (no LoRA).
    quantize = "none" / "int8" / "int4" (节省 GPU 内存, 推理质量略降).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_config = None
    if quantize in ("int4", "int8"):
        from transformers import BitsAndBytesConfig
        if quantize == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        print(f"  quantization: {quantize}")
        model = AutoModelForCausalLM.from_pretrained(
            base_path, quantization_config=quant_config, trust_remote_code=True, device_map="cuda:0",
        )
    else:
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch_dtype, trust_remote_code=True, device_map="cuda:0",
        )
    if adapter_path and adapter_path not in ("", "none", "None"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  loaded LoRA adapter: {adapter_path}")
    else:
        print("  no adapter (base model only)")
    model.eval()
    return model, tok


def generate_one(model, tok, system_prompt: str, user_prompt: str,
                  max_new_tokens: int, temperature: float) -> str:
    import torch
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(input_text, return_tensors="pt", truncation=True, max_length=8192).to("cuda:0")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    gen_ids = out[0, inputs["input_ids"].size(1):]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant",
                    choices=["v1", "v2", "v3", "v2_clean", "v2_hybrid",
                              "v2_no_pkvc_def", "v2_no_seg3_rule", "v2_no_beta_summary"],
                    default="v2")
    ap.add_argument("--n", type=int, default=20, help="0 = all 162; 20 = SAMPLE_DESIGNERS")
    ap.add_argument("--segs", default="seg1,seg2,seg3")
    ap.add_argument("--with-peer", action="store_true")
    ap.add_argument("--modules", default="pi,stance,self,caps")
    ap.add_argument("--base", default="/data/dhf/chain_critic/model/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="/data/wy/sft_v3_ckpts/stage_warmup_ep5",
                    help="LoRA adapter path; 默认 clean v3 5ep best")
    ap.add_argument("--quantize", choices=["none", "int8", "int4"], default="none",
                    help="量化加载. int4 ~5GB / int8 ~10GB / none ~14GB. zombie 占 GPU 时用 int4")
    ap.add_argument("--max-new-tokens", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit; debug 用")
    args = ap.parse_args()

    modules = {m.strip() for m in args.modules.split(",") if m.strip()}
    model_tag = "BASE" if (not args.adapter or args.adapter in ("", "none", "None")) else "SFT"
    baseline_label = f"{model_tag}_{args.variant}"
    if modules != {"pi", "stance", "self", "caps"}:
        baseline_label += f"_mods[{','.join(sorted(modules)) if modules else 'none'}]"
    if args.with_peer:
        baseline_label += "_peer"
    print(f"[config] variant={args.variant}, n={args.n}, modules={modules}, "
          f"with_peer={args.with_peer}, adapter={args.adapter}, label={baseline_label}")

    print("[load] data sources...")
    pool = load_caps_pool()
    pi = load_pi_5(Path("data_v3/users.csv"))
    scores = load_designer_scores(Path("data_v3/scale_responses.csv"))
    self_q1q6 = load_self_q1q6(Path("data_v3/task_responses.csv"))
    quads = load_caps_quadruples()
    print(f"  {len(pi)} pi, {len(scores)} scores, {len(self_q1q6)} self, {len(quads)} CAPS quads")

    if args.n == 0:
        target_designers = sorted([
            did for did in pi
            if did in scores and did in pool.designers
            and any(pool.designers[did].has_scenario(f"Q7_{s}") for s in ("seg1", "seg2", "seg3"))
        ])
    else:
        target_designers = SAMPLE_DESIGNERS
    print(f"  {len(target_designers)} target designers")

    jobs = build_jobs(args, pool, pi, scores, self_q1q6, quads, target_designers)
    if args.limit > 0:
        jobs = jobs[: args.limit]
    print(f"[build] {len(jobs)} inference jobs")
    if not jobs:
        print("no jobs, exit")
        return 0
    avg_chars = sum(len(j["prompt"]) for j in jobs) / len(jobs)
    print(f"  avg prompt chars: {avg_chars:.0f}")

    print(f"[load] Qwen base + LoRA adapter (quantize={args.quantize})...")
    model, tok = load_qwen_lora(args.base, args.adapter, quantize=args.quantize)
    print("[load] done")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    with out_path.open("w", encoding="utf-8") as f:
        for i, job in enumerate(jobs):
            seg = job["seg"]
            schema = build_seg_schema(seg)
            aug_system = augment_system_with_schema(job["system_prompt"], schema)
            try:
                raw = generate_one(
                    model, tok, aug_system, job["prompt"],
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                )
                parsed, err = parse_json_robust(raw)
                ok = parsed is not None
                rec = {
                    "ok": ok,
                    "baseline": baseline_label,
                    "designer_id": job["designer_id"],
                    "scenario": job["scenario"],
                    "raw": raw,
                }
                if ok:
                    rec.update(parsed)
                else:
                    rec["error"] = f"parse_err: {err}"
                # Attach GT
                did = job["designer_id"]
                if did in pool.designers:
                    seg_obj = pool.designers[did].get_scenario(job["scenario"])
                    if seg_obj:
                        rec["ground_truth_nl"] = seg_obj.text
                        rec["ground_truth_pkvc"] = {
                            "dominant": seg_obj.dominant_factor,
                            "sequence": seg_obj.state_sequence,
                            "criterion": seg_obj.criterion,
                            "confidence": seg_obj.confidence,
                        }
            except Exception as e:
                rec = {
                    "ok": False, "baseline": baseline_label,
                    "designer_id": job["designer_id"], "scenario": job["scenario"],
                    "error": f"{type(e).__name__}: {e}",
                }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            results.append(rec)
            # progress
            if (i + 1) % 10 == 0 or (i + 1) == len(jobs):
                el = time.time() - t0
                n_ok = sum(1 for r in results if r.get("ok"))
                rate = (i + 1) / el if el > 0 else 0
                eta = (len(jobs) - i - 1) / rate if rate > 0 else 0
                print(f"  {i + 1}/{len(jobs)} | ok={n_ok} | {el:.0f}s elapsed | {rate:.2f} jobs/s | ETA {eta:.0f}s")

    el = time.time() - t0
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\n[done] {n_ok}/{len(results)} ok in {el:.0f}s ({el/60:.1f} min)")
    print(f"[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
