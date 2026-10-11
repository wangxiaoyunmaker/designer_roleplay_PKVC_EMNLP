"""Paper-ready Tier 1 评测报告渲染.

跟 ``runner.render_report`` 的 JSON dump 不同, 这里输出 EMNLP 表格直接可贴的 markdown:

  §1  Phase 0 (Test A 数据侧 + Test B 模型侧)
  §2  B1 EFA construct validation (PKVC 4 因子)
  §3  主指标 A1/A2/A3 × 4 档 RAG 模式 (主表, brief §5.5)
  §4  Baseline 矩阵 (B0/B0'/B3/B4 vs main)
  §5  Q8 subgroup (metacognition stratification)
  §6  §5.5 D-14 mapping main-claim 升级判读
  §7  Caveats + draft 标记

不假设任何指标都有, missing 安全降级为 "—" 并提示 reason.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger("eval.report")


def _fmt(v: Any, fmt: str = ".3f") -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and v != v:    # NaN
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    try:
        return format(v, fmt)
    except (TypeError, ValueError):
        return str(v)


def _as_dict(obj: Any) -> dict:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {}


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def render_phase0_section(phase0: dict[str, Any]) -> str:
    """§1 Phase 0 Test A (数据) + Test B (模型)."""
    lines = ["## §1 Phase 0 sanity (framing viability)", ""]
    ta = _as_dict(phase0.get("test_a"))
    tb = _as_dict(phase0.get("test_b"))

    lines.append("**Test A — 数据侧 cluster 行为异质性**")
    lines.append("")
    if not ta:
        lines.append("> ✗ skipped or empty input.")
    else:
        lines.append("| metric | value | PASS? |")
        lines.append("|---|---|---|")
        lines.append(f"| ratio (between / within) | {_fmt(ta.get('mean_ratio'))} | {'✓ > 1' if (ta.get('mean_ratio') or 0) > 1 else '✗ ≤ 1'} |")
        lines.append(f"| Cramér's V (mean) | {_fmt(ta.get('cramers_v_mean'))} | {'✓ > 0.1' if (ta.get('cramers_v_mean') or 0) > 0.1 else '✗ ≤ 0.1'} |")
        lines.append(f"| permutation p | {_fmt(ta.get('permutation_p_value'), '.4g')} | {'✓ < 0.05' if (ta.get('permutation_p_value') or 1) < 0.05 else '✗ ≥ 0.05'} |")
        lines.append(f"| n_perm | {_fmt(ta.get('n_permutations'), '.0f')} | — |")
        lines.append(f"| **overall** | — | **{'✓ PASS' if ta.get('passes') else '✗ FAIL'}** |")
        if ta.get("explanation"):
            lines.append("")
            lines.append(f"> {ta['explanation']}")
    lines.append("")
    lines.append("**Test B — 模型侧 trajectory → cluster invariance**")
    lines.append("")
    if not tb or tb.get("status"):
        lines.append(f"> ✗ {tb.get('status', 'no result')} (skipped 多半是 SFT ckpt / predictions 缺).")
    else:
        chance = tb.get("chance_level", 0)
        acc = tb.get("accuracy")
        lines.append("| metric | value | chance | PASS? |")
        lines.append("|---|---|---|---|")
        lines.append(f"| classifier acc | {_fmt(acc)} | {_fmt(chance)} | {'✓ > 50%' if (acc or 0) > 0.5 else '✗ ≤ 50%'} |")
        lines.append(f"| n_samples | {_fmt(tb.get('n_samples'), '.0f')} | — | — |")
        lines.append(f"| n_clusters | {_fmt(tb.get('n_clusters'), '.0f')} | — | — |")
        if tb.get("explanation"):
            lines.append("")
            lines.append(f"> {tb['explanation']}")
    lines.append("")
    return "\n".join(lines)


def render_validation_section(validation_md: str | None) -> str:
    """§2 B1 EFA validation; 直接嵌入 ``validation.render_b1_report`` 的输出."""
    lines = ["## §2 B1 PKVC construct validation (EFA + Cronbach α + KMO + Bartlett)", ""]
    if not validation_md:
        lines.append("> skipped.")
    else:
        lines.append(validation_md.strip())
    lines.append("")
    return "\n".join(lines)


def render_main_metric_table(
    metrics_by_mode: dict[str, dict[str, float]],   # mode → {A1_weighted, A2_js_mean, A3_model_acc, A3_floor_acc, n}
    rag_mode_order: tuple[str, ...] = ("no_rag", "cosine", "og_rag", "og_rag+modulation"),
) -> str:
    """§3 主指标 × 4 档 RAG 模式表 (paper main table candidate)."""
    lines = ["## §3 主指标 × RAG mode (主方向 q1q6 → Q7 跨场景)", ""]
    lines.append("**指标解读**: A1↑ 越高越好 (trajectory match) · A2↓ 越低越好 (per-cluster JS divergence) · A3↑ 越高越好 (individual accuracy)")
    lines.append("")
    cols = ["RAG mode", "n", "A1_weighted ↑", "A1_factor ↓", "A1_sadr ↑", "A2_js_mean ↓", "A3_model ↑", "A3_floor ↑"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for mode in rag_mode_order:
        d = metrics_by_mode.get(mode)
        if not d:
            row = [f"`{mode}`", "—", "—", "—", "—", "—", "—", "—"]
        else:
            row = [
                f"`{mode}`",
                _fmt(d.get("n"), ".0f"),
                _fmt(d.get("A1_weighted")),
                _fmt(d.get("A1_factor_hamming")),
                _fmt(d.get("A1_sadr_jaccard")),
                _fmt(d.get("A2_js_mean")),
                _fmt(d.get("A3_model_acc")),
                _fmt(d.get("A3_floor_acc")),
            ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def render_baselines_section(baselines: dict[str, Any]) -> str:
    """§4 Baseline 矩阵 (B0/B0'/B3/B4)."""
    lines = ["## §4 Baseline 矩阵", ""]
    lines.append("| baseline | n_predictions | status | method |")
    lines.append("|---|---|---|---|")
    for name in sorted(baselines.keys()):
        b = baselines[name]
        if isinstance(b, dict):
            n = b.get("n_predictions")
            status = b.get("status") or ("ok" if n else b.get("error", "—"))
            sample = (b.get("sample") or [{}])[0]
            method = sample.get("method", "—") if isinstance(sample, dict) else "—"
            lines.append(f"| {name} | {_fmt(n, '.0f')} | {status} | {method} |")
        else:
            lines.append(f"| {name} | — | {b} | — |")
    lines.append("")
    lines.append("> B0/B0' offline 用 random / cluster-majority fallback; paper 用真 LLM API 重跑替换.")
    lines.append("")
    return "\n".join(lines)


def render_q8_section(q8_md: str | None) -> str:
    """§5 Q8 metacognition subgroup (借 metrics.render_subgroup_markdown 渲染)."""
    lines = ["## §5 Q8 metacognition subgroup", ""]
    if not q8_md:
        lines.append("> skipped (extractor / predictions 未就位).")
    else:
        # strip 内层 H2 避免双重标题
        cleaned = q8_md.replace("## Q8 子群评测报告", "").lstrip()
        lines.append(cleaned)
    lines.append("")
    return "\n".join(lines)


def render_mapping_judgment(nway: Any) -> str:
    """§6 §5.5 D-14 mapping main-claim 升级判读."""
    lines = ["## §6 §5.5 mapping main-claim 升级判读 (D-14)", ""]
    if nway is None:
        lines.append("> ✗ skipped (RAG ablation 未跑).")
        lines.append("")
        return "\n".join(lines)
    d = _as_dict(nway)
    lines.append(f"- metric: `{d.get('metric_name', '?')}`")
    lines.append(f"- reference mode: `{d.get('reference_mode', '?')}`")
    lines.append(f"- ranking (best → worst):")
    for m, s in d.get("ranking", []):
        lines.append(f"  - `{m}`: {_fmt(s)}")
    lines.append(f"- og_rag+modulation − og_rag = {_fmt(d.get('mapping_delta'))}")
    lines.append(f"- **rag_helps_at_all** (max(rag) − no_rag > 1%): {_fmt(d.get('rag_helps_at_all'))}")
    lines.append(f"- **mapping_promotes_main_claim** (Δ > 1%): {_fmt(d.get('mapping_promotes_main_claim'))}")
    lines.append("")
    if d.get("mapping_promotes_main_claim"):
        lines.append("> ✓ **建议**: §5.5 升级 mapping triple 为 main contribution; 跟学姐 / 老师沟通 D-17 (一作分工).")
    elif d.get("rag_helps_at_all"):
        lines.append("> ↻ peer RAG 有效但 mapping 增益 < 1%, 保持 og_rag 为 main; mapping 留 ablation 表.")
    else:
        lines.append("> ✗ RAG 整体没明显增益, 重新审视 framing (peer NL 质量 / similarity metric / k=3 是否合理).")
    lines.append("")
    return "\n".join(lines)


def render_caveats(extra: list[str] | None = None) -> str:
    lines = ["## §7 Caveats", ""]
    lines.append("- draft 状态. 主要 Tier 1; Tier 2/3 (SM-abl / B2 CFA / D3 反向 / D4 memorization) 另外跑.")
    lines.append("- 200 入袋前所有数字 underpowered, 投稿前 200 重跑.")
    lines.append("- B0/B0' 在 fallback 模式下不代表真 LLM 性能; 投稿前 API 重跑.")
    lines.append("- Phase 0 Test B 需要 SFT ckpt + 推理输出 jsonl 才会生效, 否则 skip.")
    lines.append("- §5.5 D-14 判读默认 1% threshold, 跟 GPT review 商榷可改.")
    if extra:
        for c in extra:
            lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level assembler
# ---------------------------------------------------------------------------

def render_tier1_report(
    *,
    phase0: dict[str, Any] | None = None,
    validation_md: str | None = None,
    metrics_by_mode: dict[str, dict[str, float]] | None = None,
    baselines: dict[str, Any] | None = None,
    q8_md: str | None = None,
    rag_nway_result: Any | None = None,
    direction: str = "q1q6",
    rag_mode_order: tuple[str, ...] = ("no_rag", "cosine", "og_rag", "og_rag+modulation"),
    extra_caveats: list[str] | None = None,
) -> str:
    """投稿用 Tier 1 完整报告 (markdown). 全部参数可选, missing 安全降级."""
    timestamp = datetime.now().isoformat(timespec="minutes")
    blocks = [
        f"# Tier 1 评测报告 [draft] · direction={direction} · {timestamp}",
        "",
        "> 此报告 = Phase 0 sanity + B1 EFA construct + 主指标 (4 档 RAG) + baseline 矩阵 + Q8 subgroup + §5.5 D-14 判读.",
        "> v3.5.1/v3.7 brief 锁定; missing 段表示对应输入未就位 (SFT ckpt / 推理 jsonl / extractor API key).",
        "",
    ]
    if phase0:
        blocks.append(render_phase0_section(phase0))
    blocks.append(render_validation_section(validation_md))
    blocks.append(render_main_metric_table(metrics_by_mode or {}, rag_mode_order))
    if baselines:
        blocks.append(render_baselines_section(baselines))
    blocks.append(render_q8_section(q8_md))
    blocks.append(render_mapping_judgment(rag_nway_result))
    blocks.append(render_caveats(extra_caveats))
    return "\n".join(blocks)
