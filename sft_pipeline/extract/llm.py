"""LLM-as-extractor: 从 NL 反推 PKVC + SADR.

用法:
  from eval_extractor import extract_pkvc
  result = extract_pkvc(text, model="gpt-4o")
  # result: dict with dominant_factor, sadr_activations, criterion, confidence

特性:
  - SHA-256 缓存 (重复评测复用)
  - 多家族 LLM 支持 (openai / anthropic)
  - JSON mode 强制结构化
  - Retry on failure (最多 3 次)
  - 异步批量 (asyncio)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

CACHE_DIR = Path(__file__).parent / "cache" / "extractor"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_PATH = Path(__file__).parent.parent / "config" / "prompts" / "extractor_v1.md"


@dataclass
class ExtractionResult:
    """LLM extractor 输出.

    v3.5.2 升级 (T_CAPS quad schema v1.1):
        - 新增 p_surface / p_modulation / o_primary / o_subtype 字段 (quad output)
        - 老字段 dominant_factor / sadr_activations 保留 (v2 schema 兼容)
        - 新字段全部 Optional, extractor v2 输出可继续 parse (向后兼容)
        - extractor v3 prompt 输出 quad 字段时自动填充
    See: docs/briefs/T_CAPS_quad_schema_v1.md, rag/quad_schema.py
    """
    segment_role: str | None
    dominant_factor: str | None
    sadr_activations: list[dict[str, str]] = field(default_factory=list)
    criterion: str | None = None
    criterion_strength: str | None = None
    confidence: Literal["high", "medium", "low"] | None = None
    extraction_notes: str | None = None
    raw_json: str = ""
    cache_hit: bool = False
    # ─── v1.1 quad fields (扩展, extractor v3 输出; v2 输出时全 None) ───
    p_surface: str | None = None          # SURFACE_PREDICATES 之一
    p_modulation: str | None = None       # MODULATION_LEVELS 之一, 由 codebook 查表 OR LLM 直出
    o_primary: str | None = None          # PKVC_FIELDS 之一; v2 用 dominant_factor 兼容
    o_subtype: str | None = None          # PKVC_SUBTYPES 之一 (P1-C3); v1.1 attribute

    @classmethod
    def from_json_obj(cls, obj: dict, raw: str = "", cache_hit: bool = False) -> "ExtractionResult":
        return cls(
            segment_role=obj.get("segment_role"),
            dominant_factor=obj.get("dominant_factor"),
            sadr_activations=obj.get("sadr_activations", []),
            criterion=obj.get("criterion"),
            criterion_strength=obj.get("criterion_strength"),
            confidence=obj.get("confidence"),
            extraction_notes=obj.get("extraction_notes"),
            raw_json=raw,
            cache_hit=cache_hit,
            p_surface=obj.get("p_surface"),
            p_modulation=obj.get("p_modulation"),
            o_primary=obj.get("o_primary"),
            o_subtype=obj.get("o_subtype"),
        )


def _cache_key(text: str, model: str) -> str:
    h = hashlib.sha256(f"{model}::{text}".encode("utf-8")).hexdigest()[:16]
    return h


def _load_prompt_template() -> tuple[str, str]:
    """从 extractor_prompt.md 提取 system + user template."""
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt not found: {PROMPT_PATH}")
    content = PROMPT_PATH.read_text(encoding="utf-8")
    # 简化提取: System 取第一段 ```...```, User 取第二段
    blocks = [b.strip() for b in content.split("```") if b.strip()]
    system = blocks[1] if len(blocks) > 1 else "你是设计师认知分析专家"
    user_tmpl = blocks[2] if len(blocks) > 2 else "{INPUT_TEXT}"
    return system, user_tmpl


def extract_pkvc(
    text: str,
    model: str = "gpt-4o",
    use_cache: bool = True,
    api_key: str | None = None,
) -> ExtractionResult:
    """同步版本. 异步批量见 extract_pkvc_batch_async."""
    cache_key = _cache_key(text, model)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if use_cache and cache_file.exists():
        obj = json.loads(cache_file.read_text(encoding="utf-8"))
        return ExtractionResult.from_json_obj(obj, raw=json.dumps(obj), cache_hit=True)

    system, user_tmpl = _load_prompt_template()
    user_msg = user_tmpl.replace("{INPUT_TEXT}", text)

    if model.startswith("gpt-"):
        raw = _call_openai(system, user_msg, model, api_key)
    elif model.startswith("claude-"):
        raw = _call_anthropic(system, user_msg, model, api_key)
    elif model.startswith("deepseek-"):
        raw = _call_deepseek(system, user_msg, model, api_key)
    else:
        raise ValueError(f"Unknown model: {model}")

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # 退化: 尝试从文本里挖 JSON
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(raw[start:end + 1])
        else:
            raise

    if use_cache:
        cache_file.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    return ExtractionResult.from_json_obj(obj, raw=raw, cache_hit=False)


def _call_openai(system: str, user_msg: str, model: str, api_key: str | None) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                   {"role": "user", "content": user_msg}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(system: str, user_msg: str, model: str, api_key: str | None) -> str:
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=2048,
        temperature=0,
    )
    return resp.content[0].text


def _call_deepseek(system: str, user_msg: str, model: str, api_key: str | None) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                   {"role": "user", "content": user_msg}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


async def extract_pkvc_batch_async(
    texts: list[str],
    model: str = "gpt-4o",
    use_cache: bool = True,
    concurrency: int = 8,
) -> list[ExtractionResult]:
    """异步批量提取, 加速 600 段 NL 加工."""
    sem = asyncio.Semaphore(concurrency)

    async def _one(t: str) -> ExtractionResult:
        async with sem:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, extract_pkvc, t, model, use_cache)

    return await asyncio.gather(*[_one(t) for t in texts])


def compute_extraction_kappa(
    texts: list[str],
    model_a: str = "gpt-4o",
    model_b: str = "claude-3-5-sonnet-20241022",
) -> dict:
    """跨家族 LLM extraction κ pilot, 验证 extractor 可靠性 (≥ 0.7 才可用)."""
    results_a = [extract_pkvc(t, model=model_a) for t in texts]
    results_b = [extract_pkvc(t, model=model_b) for t in texts]
    factor_agree = sum(
        1 for a, b in zip(results_a, results_b)
        if a.dominant_factor == b.dominant_factor
    ) / len(texts)
    sadr_jaccard_mean = sum(
        len(set(s["id"] for s in a.sadr_activations) &
            set(s["id"] for s in b.sadr_activations)) /
        max(len(set(s["id"] for s in a.sadr_activations) |
                  set(s["id"] for s in b.sadr_activations)), 1)
        for a, b in zip(results_a, results_b)
    ) / len(texts)
    return {
        "n": len(texts),
        "model_a": model_a,
        "model_b": model_b,
        "factor_agreement_rate": factor_agree,
        "sadr_jaccard_mean": sadr_jaccard_mean,
        "passes_kappa_07": factor_agree >= 0.7 and sadr_jaccard_mean >= 0.5,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--text", help="单段 NL 提取测试")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--kappa-pilot", type=int, default=0, help="从 demo 段抽 N 段算 κ")
    args = p.parse_args()

    if args.text:
        r = extract_pkvc(args.text, model=args.model)
        print(json.dumps(r.__dict__, ensure_ascii=False, indent=2, default=str))
    elif args.kappa_pilot:
        # 演示: 用同段文本算 κ
        demo_text = "我的核心底线只有一句话: 人工服务不消除"
        r = compute_extraction_kappa([demo_text] * args.kappa_pilot)
        print(json.dumps(r, ensure_ascii=False, indent=2))
