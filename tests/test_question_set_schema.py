# -*- coding: utf-8 -*-
"""文生图题库 JSON 必填字段与结构校验。"""

import json
import pytest
from pathlib import Path


REQUIRED_KEYS = ["question_id", "dimension", "question_text"]


def _validate_question(q: dict, index: int) -> list:
    errors = []
    for key in REQUIRED_KEYS:
        if key not in q:
            errors.append(f"题目 #{index}: 缺少必填字段 '{key}'")
        elif key == "dimension" and q.get(key) != "text_to_image":
            errors.append(f"题目 #{index}: dimension 应为 'text_to_image'，当前为 '{q.get(key)}'")
    if not isinstance(q.get("question_text"), str) or not (q.get("question_text") or "").strip():
        errors.append(f"题目 #{index}: question_text 必须为非空字符串")
    return errors


def validate_questions_list(data: list) -> list:
    errors = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"题目 #{i}: 应为对象")
            continue
        errors.extend(_validate_question(item, i))
    return errors


class TestTextToImageQuestionSetSchema:
    """文生图题库 schema 校验"""

    def test_valid_sample(self):
        base = Path(__file__).resolve().parent.parent
        path = base / "livesecbench" / "question_set" / "text_to_image" / "sample.json"
        if not path.exists():
            pytest.skip("sample.json 不存在")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        errors = validate_questions_list(data)
        assert not errors, errors

    def test_missing_required_field(self):
        q = {"question_id": "x", "dimension": "text_to_image"}
        errors = _validate_question(q, 0)
        assert any("question_text" in e for e in errors)

    def test_wrong_dimension(self):
        q = {"question_id": "x", "dimension": "other", "question_text": "a prompt"}
        errors = _validate_question(q, 0)
        assert any("text_to_image" in e for e in errors)

    def test_empty_question_text(self):
        q = {"question_id": "x", "dimension": "text_to_image", "question_text": ""}
        errors = _validate_question(q, 0)
        assert any("question_text" in e for e in errors)
