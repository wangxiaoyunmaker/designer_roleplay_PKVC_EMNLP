"""Unit tests for per-seg structured output schema (3 schemas).

Run: .venv/bin/python -m pytest sft_pipeline/tests/test_seg_schema.py -v
"""
import sys
from pathlib import Path

import jsonschema
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from run_3_baselines_q7 import build_seg_schema


def _valid_base(seg):
    common = {
        "confidence": "high",
        "reasoning_notes": "ok",
        "nl_answer": "这是一个测试回答, 我觉得很简单, 直觉上我会..." * 5,
    }
    return common


# --- seg1 schema ---

def test_seg1_valid_single_letter():
    schema = build_seg_schema("seg1")
    obj = {
        "dominant_factor": "V",
        "state_sequence": ["V"],
        "criterion": None,
        **_valid_base("seg1"),
    }
    jsonschema.Draft202012Validator(schema).validate(obj)


def test_seg1_rejects_path():
    schema = build_seg_schema("seg1")
    obj = {
        "dominant_factor": "V→K",                # seg1 禁 X→Y
        "state_sequence": ["V"], "criterion": None,
        **_valid_base("seg1"),
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(obj)


def test_seg1_rejects_long_sequence():
    schema = build_seg_schema("seg1")
    obj = {
        "dominant_factor": "V",
        "state_sequence": ["V","K","P","V"],     # length 4 > maxItems 3
        "criterion": None,
        **_valid_base("seg1"),
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(obj)


def test_seg1_allows_null_criterion():
    schema = build_seg_schema("seg1")
    obj = {"dominant_factor": "K", "state_sequence": ["K"], "criterion": None, **_valid_base("seg1")}
    jsonschema.Draft202012Validator(schema).validate(obj)


# --- seg2 schema ---

def test_seg2_requires_multistate():
    schema = build_seg_schema("seg2")
    obj = {
        "dominant_factor": "V",
        "state_sequence": ["V"],                 # length 1 < minItems 2
        "criterion": None,
        **_valid_base("seg2"),
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(obj)


def test_seg2_valid_chain():
    schema = build_seg_schema("seg2")
    obj = {
        "dominant_factor": "V→K",
        "state_sequence": ["V", "K", "P", "V"],
        "criterion": None,
        **_valid_base("seg2"),
    }
    jsonschema.Draft202012Validator(schema).validate(obj)


def test_seg2_allows_path():
    schema = build_seg_schema("seg2")
    obj = {
        "dominant_factor": "K→V",
        "state_sequence": ["K", "V"],
        "criterion": "用户优先",
        **_valid_base("seg2"),
    }
    jsonschema.Draft202012Validator(schema).validate(obj)


def test_seg2_rejects_too_long_sequence():
    schema = build_seg_schema("seg2")
    obj = {
        "dominant_factor": "V",
        "state_sequence": ["V","K","P","V","K","P","V","K"],   # length 8 > 7
        "criterion": None,
        **_valid_base("seg2"),
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(obj)


# --- seg3 schema ---

def test_seg3_requires_criterion():
    schema = build_seg_schema("seg3")
    obj = {
        "dominant_factor": "V",
        "state_sequence": ["V"],
        "criterion": None,                        # seg3 必须非 null
        **_valid_base("seg3"),
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(obj)


def test_seg3_rejects_path():
    schema = build_seg_schema("seg3")
    obj = {
        "dominant_factor": "V→K",
        "state_sequence": ["V"],
        "criterion": "用户体验",
        **_valid_base("seg3"),
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(obj)


def test_seg3_valid_with_criterion():
    schema = build_seg_schema("seg3")
    obj = {
        "dominant_factor": "V",
        "state_sequence": ["V"],
        "criterion": "用户体验不可妥协",
        **_valid_base("seg3"),
    }
    jsonschema.Draft202012Validator(schema).validate(obj)


def test_seg3_criterion_too_long():
    schema = build_seg_schema("seg3")
    obj = {
        "dominant_factor": "V",
        "state_sequence": ["V"],
        "criterion": "这是一个超过二十五个字符的非常长的底线短句, 应该被 schema 拒绝",
        **_valid_base("seg3"),
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(obj)


# --- common ---

def test_all_schemas_reject_extra_fields():
    for seg in ("seg1", "seg2", "seg3"):
        schema = build_seg_schema(seg)
        obj = {
            "dominant_factor": "V",
            "state_sequence": ["V"] if seg != "seg2" else ["V", "K"],
            "criterion": "底线" if seg == "seg3" else None,
            **_valid_base(seg),
            "extra_field": "should fail",
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(schema).validate(obj)


def test_invalid_seg_name_raises():
    with pytest.raises(ValueError):
        build_seg_schema("seg99")
