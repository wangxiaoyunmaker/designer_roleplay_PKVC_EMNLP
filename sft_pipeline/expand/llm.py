"""单条 LLM 调用 + 文件 cache (Q1-Q6 扩写专用).

跟 extract/llm.py 的 cache + retry 模式同构, 但: (a) prompt 来自 expand/prompt.py
而不是 extractor template; (b) 返回 raw JSON 字符串, 不直接 parse (parse 交给
expand/parser.py 因为要传入 candidate 做 evidence_quote 校验).

只支持 gpt-4o 系列 (默认), 用 yunwu.ai 中转 (OPENAI_API_KEY 由 user provide).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable

logger = logging.getLogger("expand.llm")

CACHE_DIR = Path(__file__).parent / "cache" / "expander"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(system: str, user: str, model: str) -> str:
    payload = f"{model}::{system}::{user}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def call_chat(
    system: str,
    user: str,
    model: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
    use_cache: bool = True,
    temperature: float = 0.3,
    response_format: dict | None = None,
) -> tuple[str, bool]:
    """返回 (raw_json_str, cache_hit). raw_json_str 是 LLM 的原始 string,
    解析交给 parser.parse_and_validate.

    response_format: 默认 None → 用 {"type": "json_object"} (旧行为).
                     传入 strict json_schema dict (从 extract/schema.py get_schema()
                     来) → 100% enforce 字段名 + 类型 + 闭集, 砍 prompt 自律铁律.
    """
    # cache key 含 response_format 模式 (不同 schema 不同 cache)
    rf_tag = ""
    if response_format and response_format.get("type") == "json_schema":
        rf_tag = f"::schema:{response_format['json_schema']['name']}"
    ck = _cache_key(system, user, model + rf_tag)
    cache_file = CACHE_DIR / f"{ck}.json"
    if use_cache and cache_file.exists():
        raw = cache_file.read_text(encoding="utf-8")
        return raw, True

    if model.startswith("gpt-"):
        raw = _call_openai(system, user, model, api_key, base_url, temperature, response_format)
    elif model.startswith("claude-"):
        raw = _call_anthropic(system, user, model, api_key, temperature)
    elif model.startswith("deepseek-"):
        raw = _call_deepseek(system, user, model, api_key, temperature)
    else:
        raise ValueError(f"Unknown model family: {model}")

    if use_cache:
        cache_file.write_text(raw, encoding="utf-8")
    return raw, False


def _call_openai(
    system: str, user: str, model: str,
    api_key: str | None, base_url: str | None, temperature: float,
    response_format: dict | None = None,
) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),  # yunwu.ai 中转
    )
    rf = response_format or {"type": "json_object"}
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        response_format=rf,
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(
    system: str, user: str, model: str, api_key: str | None, temperature: float,
) -> str:
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=1024,
        temperature=temperature,
    )
    return resp.content[0].text


def _call_deepseek(
    system: str, user: str, model: str, api_key: str | None, temperature: float,
) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Mock LLM (offline smoke test)
# ---------------------------------------------------------------------------

def mock_call_chat_factory(seed_template: str | None = None) -> Callable[..., tuple[str, bool]]:
    """返回一个跟 call_chat 同签名的 mock 函数, 不调网络.

    用于 smoke test: 从 user_msg 提取 SEED + Q7_B1 第一句, 装配出符合 schema
    的扩写, 直接走 parse + validate, 跑通端到端.
    """
    import re

    def _mock(system: str, user: str, model: str = "gpt-4o", **kwargs) -> tuple[str, bool]:
        # 从 user prompt 抠 seed 和 Q7_B1 第一句
        seed_match = re.search(r'"([^"]{0,30})"\s*\n\n\(原始', user)
        seed = seed_match.group(1) if seed_match else "直觉"
        b1_match = re.search(r"\[第一反应 B1\]\n(.+?)(?:\n\n|\n\[)", user, re.DOTALL)
        b1 = b1_match.group(1).strip() if b1_match else ""
        # 取 B1 前 25 字作 evidence_quote
        quote = b1[:20] if len(b1) >= 8 else "我的核心权衡逻辑是"
        # 装一个 50-100 字扩写
        expanded = (
            f"我当时的脑子里就是「{seed}」两个字先跑出来. 我习惯先靠场景画面感判断, "
            f"再用数据回查. 这次也是先有感觉再选的, 后面想想跟我对老城区那种"
            f"场所感的理解是一致的."
        )
        # 卡到 ≤100
        if len(expanded) > 100:
            expanded = expanded[:99] + "."
        if len(expanded) < 50:
            expanded += "我大概是凭这个走的."
        obj = {
            "expanded_nl": expanded,
            "inferred_factor": "K",
            "inferred_subtype": "K1",
            "evidence_quote": quote[:30],
            "evidence_source": "Q7_B1",
            "confidence": "high",
            "reason_if_low": "",
        }
        return json.dumps(obj, ensure_ascii=False), False

    return _mock
