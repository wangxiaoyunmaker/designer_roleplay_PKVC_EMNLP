"""infer_v2.py — Q7.seg1/2/3 inference per (test fold user × baseline).

Framework v1.1 §6 cognitive_function 跨场景迁移:
  Q1 → Q7.seg1 (first_reaction)
  Q2 → Q7.seg2 (weighing)
  Q3-Q6 → Q7.seg3 (bottom_line)

每个 test fold user 跑 3 个 seg × N baseline:
  - prompt = SCENARIO (evaluation, 老城区超市) + question + routing_format
            + 同样的 profile / own_responses / own_quads / peer_raw_nl (baseline 决定)
  - completion = model.generate (parse 出 [认知路由标签] + [NL 回答])
  - output: gen/v2/fold_{k}/{baseline}/{user_id}__seg{1,2,3}.json

CLI:
    python -m sft_pipeline.infer_v2 \\
        --config sft_pipeline/config/config_v2.yaml \\
        --fold 0 --baselines B1 B5
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 复用 prep_v2 的装配 helpers (避免重复维护 prompt schema)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sft_pipeline.data.prep_v2 import (
    Q_TO_FUNCTION, ROUTING_FORMAT_INSTRUCTIONS, COGNITIVE_FUNCTION_QUESTIONS,
    SYSTEM_MESSAGE, PromptConfig,
    build_profile_matrix, numerical_to_polarity, polarity_to_nl,
    compute_factor_scores, parse_task_responses,
    load_q1q6_option_labels, load_q2_training_labels,
    load_q1q6_expansion, load_q7_segments, retrieve_peers,
    fmt_profile_section, fmt_own_responses_section,
    fmt_own_quads_section_fallback, fmt_peer_raw_nl_section, fmt_scenario_section,
    demographic_str,
)

logger = logging.getLogger("infer_v2")


# Q7.segN 对应的 cognitive function (跟 Q1/Q2/Q3-Q6 训练 function 对齐)
SEG_TO_QID_HINT = {"seg1": "Q1", "seg2": "Q2", "seg3": "Q3"}  # for routing_format lookup


def build_q7_seg_prompt(
    cfg_baseline: PromptConfig,
    polarity_nl: str, demographics: str,
    user_resp: dict[str, dict], q2_entry: dict | None,
    expansion: dict[tuple[str, str], str],
    option_labels: dict[tuple[str, str], dict],
    peers: list[dict],
    scenario_text_eval: str,
    seg: str,
    uid: str,
) -> tuple[str, str]:
    """Q7.seg1/2/3 推理 prompt 装配. 跟训练 prompt 严格同 schema.
    own_responses / own_quads mask 不需要 (Q7 跟 Q1-Q6 不冲突).
    """
    func = {"seg1": "first_reaction", "seg2": "weighing", "seg3": "bottom_line"}[seg]
    sections = []
    if cfg_baseline.profile:
        sections.append(fmt_profile_section(polarity_nl, demographics))
    if cfg_baseline.own_responses:
        # Q7 推理时 own_responses 全开 (Q1-Q6 都给, 不 mask)
        sections.append(fmt_own_responses_section(
            user_resp, q2_entry, expansion, option_labels, uid, mask_qid="__none__"
        ))
    if cfg_baseline.own_quads:
        s = fmt_own_quads_section_fallback()
        if s:
            sections.append(s)
    if cfg_baseline.peer_raw_nl:
        s = fmt_peer_raw_nl_section(peers)
        if s:
            sections.append(s)
    # scenario + question + routing_format (跟训练同结构, function 决定)
    scenario_block = (
        "╔═══ 场景 ═══╗\n\n"
        f"{scenario_text_eval}\n\n"
        "╚═══════════╝\n\n"
        f"{COGNITIVE_FUNCTION_QUESTIONS[func]}\n\n"
        "[认知路由标签格式 (本次必须遵守)]\n"
        f"{ROUTING_FORMAT_INSTRUCTIONS[func]}"
    )
    sections.append(scenario_block)
    return SYSTEM_MESSAGE, "\n\n".join(sections)


# ─────────────────────────────────────────────────────────────────────
# Parse generated completion
# ─────────────────────────────────────────────────────────────────────

_ROUTING_RE = re.compile(r"\[认知路由标签\]\s*\n(.*?)\n\s*\n\s*\[NL 回答\]\s*\n(.+?)$",
                          re.DOTALL)


def parse_completion(text: str) -> dict:
    """Extract routing + NL from generated text. Returns {routing, nl, raw, parsed_ok}."""
    m = _ROUTING_RE.search(text)
    if m:
        return {"routing": m.group(1).strip(), "nl": m.group(2).strip(),
                "raw": text, "parsed_ok": True}
    # fallback: try less-strict
    if "[NL 回答]" in text:
        parts = text.split("[NL 回答]", 1)
        head = parts[0].replace("[认知路由标签]", "").strip()
        nl = parts[1].strip()
        return {"routing": head, "nl": nl, "raw": text, "parsed_ok": False}
    return {"routing": "", "nl": text.strip(), "raw": text, "parsed_ok": False}


# ─────────────────────────────────────────────────────────────────────
# Inference loop
# ─────────────────────────────────────────────────────────────────────

def load_model(cfg: dict, adapter_path: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_cfg = cfg["model"]
    base = model_cfg.get("remote_base_model") \
        if Path(model_cfg.get("remote_base_model", "/nonexistent")).exists() \
        else model_cfg["base_model"]
    logger.info("Loading base %s + adapter %s ...", base, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def generate_once(model, tokenizer, system: str, user: str, infer_cfg: dict) -> str:
    import torch
    msgs = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=infer_cfg["max_new_tokens"],
            temperature=infer_cfg["temperature"],
            top_p=infer_cfg["top_p"],
            do_sample=infer_cfg["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run_fold(cfg: dict, fold: int, baselines: list[str]) -> None:
    paths = cfg["paths"]
    root = Path.cwd()
    adapter_path = Path(paths["ckpt_dir"]) / f"fold_{fold}"
    if not adapter_path.exists():
        logger.error("Adapter not found: %s. 跑 train_v2 先?", adapter_path)
        sys.exit(1)

    # Load all the data sources prep_v2 also uses (跟训练 prompt 同源)
    logger.info("Loading data sources for fold %d ...", fold)
    users = pd.read_csv(root / paths["users"])
    users["user_id"] = users["wechat_id"].astype(str)
    users = users.set_index("user_id")
    scales = pd.read_csv(root / paths["scale_responses"])
    tasks = pd.read_csv(root / paths["task_responses"])

    factor_df = compute_factor_scores(scales, root / paths["efa_loadings"])
    factor_df.index = factor_df.index.astype(str)
    option_labels = load_q1q6_option_labels(root / paths["q1q6_option_labels"])
    q2_labels = load_q2_training_labels(root / paths["q2_training_labels"])
    expansion = load_q1q6_expansion(root / paths["q1q6_expansion_outputs"])
    q7_segs = load_q7_segments(root / paths["q7_extracted_v8_7"])
    user_resp = parse_task_responses(tasks)

    profile_matrix = build_profile_matrix(scales)
    profile_matrix.index = profile_matrix.index.astype(str)

    # determine test fold users (按 train_v2 同样的 fold_split)
    from sft_pipeline.data.prep_v2 import trait_stratified_fold_split
    fold_of = trait_stratified_fold_split(factor_df, cfg["data"]["n_folds"],
                                            cfg["train"]["seed"])
    test_uids = [u for u, fk in fold_of.items() if fk == fold]
    train_uids = {u for u in fold_of if fold_of[u] != fold}
    logger.info("test fold %d: %d users", fold, len(test_uids))

    model, tokenizer = load_model(cfg, adapter_path)
    gen_dir = Path(paths["gen_dir"]) / f"fold_{fold}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    scen_eval = cfg["scenario_text"]["evaluation"]
    infer_cfg = cfg["infer"]

    for bname in baselines:
        b = cfg["baselines"][bname]
        pcfg = PromptConfig(profile=b["profile"], own_responses=b["own_responses"],
                            own_quads=b["own_quads"], peer_raw_nl=b["peer_raw_nl"])
        b_dir = gen_dir / bname
        b_dir.mkdir(parents=True, exist_ok=True)

        for uid in test_uids:
            if uid not in profile_matrix.index:
                continue
            polarity = numerical_to_polarity(profile_matrix.loc[uid])
            polarity_nl_str = polarity_to_nl(polarity)
            demo = demographic_str(users.loc[uid]) if uid in users.index else ""
            peers = retrieve_peers(uid, factor_df, train_uids, q7_segs, k=3)
            q2_entry = q2_labels.get(uid)

            for seg in ("seg1", "seg2", "seg3"):
                sys_msg, user_msg = build_q7_seg_prompt(
                    pcfg, polarity_nl_str, demo,
                    user_resp.get(uid, {}), q2_entry, expansion, option_labels,
                    peers, scen_eval, seg, uid,
                )
                gen_text = generate_once(model, tokenizer, sys_msg, user_msg, infer_cfg)
                parsed = parse_completion(gen_text)
                out_path = b_dir / f"{uid}__{seg}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump({
                        "user_id": uid, "fold": fold, "baseline": bname, "seg": seg,
                        "prompt_system": sys_msg, "prompt_user": user_msg,
                        **parsed,
                    }, f, ensure_ascii=False, indent=2)
                logger.info("[%s/%s] %s %s parsed_ok=%s  routing[:40]=%r",
                             bname, uid, seg,
                             "✓" if parsed["parsed_ok"] else "⚠",
                             parsed["parsed_ok"], parsed["routing"][:40])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sft_pipeline/config/config_v2.yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--baselines", nargs="+", default=["B5"])
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_fold(cfg, args.fold, args.baselines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
