"""JSON Schema for OpenAI response_format=json_schema strict mode.

用 strict schema 替代 prompt 自律铁律 (字段名严格 / 不嵌套 / 不加字段 / 不要 markdown
围栏). GPT-4o + json_schema strict mode 100% 保证字段名 + 类型 + 闭集.

参考: docs/briefs/T_CAPS_quad_schema_v1.md §配套 batch.py 改动

注意 strict mode 限制:
  - 所有 properties 字段必须 ∈ required
  - additionalProperties = False (没有 default 字段, 不允许漏字段)
  - enum 严格按列出值, 不允许变体
  - nullable 用 ["string", "null"] (anyOf 形式)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# v8 PKVC extractor schema (4 字段)
# ---------------------------------------------------------------------------
PKVC_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "dominant_factor": {
            "type": "string",
            "description": "单字母 P/K/V/C 或路径如 K→V (上限 2 段)",
            # 不用 enum 因为路径展开过多, 用 pattern 约束
            "pattern": r"^(P|K|V|C)(→(P|K|V|C))?$",
        },
        "criterion": {
            "anyOf": [{"type": "string", "maxLength": 25}, {"type": "null"}],
            "description": "≤25 字 verbatim INPUT 节选, 或 null",
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "extraction_notes": {
            "type": "string",
            "maxLength": 60,
            "description": "≤60 字一句话总结",
        },
    },
    "required": ["dominant_factor", "criterion", "confidence", "extraction_notes"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# T_CAPS quad extractor schema (5 字段, LLM 只出 p_surface/o_primary/o_subtype + meta)
# ---------------------------------------------------------------------------
P_SURFACE_VOCAB_LIST = [
    "forces_transition_to",
    "severely_overrides",
    "defiant_overrides",
    "gradually_shifts_to",
    "compliantly_follows",
    "negotiates_around",
    "weakly_inhibits",
    "bypasses",
]
O_PRIMARY_VOCAB_LIST = ["P", "K", "V", "C"]
O_SUBTYPE_VOCAB_LIST = [
    "P1", "P2", "P3",
    "K1", "K2", "K3",
    "V1", "V2", "V3",
    "C1", "C2", "C3",
]

QUAD_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "p_surface": {
            "type": "string",
            "enum": P_SURFACE_VOCAB_LIST,
            "description": "8 词闭集, designer 的认知路由谓语",
        },
        "o_primary": {
            "type": "string",
            "enum": O_PRIMARY_VOCAB_LIST,
            "description": "4 状态, dominant 路径终点",
        },
        "o_subtype": {
            "anyOf": [
                # 单值 (12 闭集), 终点 == o_subtype 自己
                {"type": "string", "enum": O_SUBTYPE_VOCAB_LIST},
                # 复合 X→Y (动态关系), 终点 Y 必须跟 o_primary 一致
                # pattern: 起点 P1-C3 + "→" + 终点 P1-C3
                {"type": "string", "pattern": "^(P[1-3]|K[1-3]|V[1-3]|C[1-3])→(P[1-3]|K[1-3]|V[1-3]|C[1-3])$"},
                {"type": "null"},
            ],
            "description": "12 cardId 单值 或 X→Y 复合 (动态关系). 终点 prefix 必须跟 o_primary 一致.",
        },
        "sadr_activations": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "pattern": "^#(P[1-3]|K[1-3]|V[1-3]|C[1-3])[+-]$"},
                    "label": {"type": "string", "maxLength": 15},
                },
                "required": ["id", "label"],
                "additionalProperties": False,
            },
            "description": "1-4 个 SADR 激活点 dict {id, label}; id 形式 #cardId极性",
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "extraction_notes": {
            "type": "string",
            "maxLength": 60,
            "description": "≤60 字, 指明 NL 哪个短语触发",
        },
    },
    "required": ["p_surface", "o_primary", "o_subtype", "sadr_activations", "confidence", "extraction_notes"],
    "additionalProperties": False,
}


def get_schema(name: str) -> dict:
    """返回 OpenAI response_format=json_schema 的完整结构.

    name ∈ {pkvc_extraction, quad_extraction}
    """
    schemas = {
        "pkvc_extraction": PKVC_EXTRACTION_SCHEMA,
        "quad_extraction": QUAD_EXTRACTION_SCHEMA,
    }
    if name not in schemas:
        raise ValueError(f"Unknown schema {name!r}, options: {list(schemas)}")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schemas[name],
            "strict": True,
        },
    }


if __name__ == "__main__":
    import json
    print("PKVC schema 字段:")
    print(json.dumps(PKVC_EXTRACTION_SCHEMA["properties"], ensure_ascii=False, indent=2)[:500])
    print()
    print("QUAD schema 字段:")
    print(json.dumps(QUAD_EXTRACTION_SCHEMA["properties"], ensure_ascii=False, indent=2)[:500])
    print()
    print("get_schema('quad_extraction') wrapper:")
    print(json.dumps(get_schema("quad_extraction"), ensure_ascii=False, indent=2)[:300])
