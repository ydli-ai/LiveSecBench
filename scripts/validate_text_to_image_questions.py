#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校验文生图题库 JSON 的必填字段与结构。
用法: python scripts/validate_text_to_image_questions.py [question_set/text_to_image]
"""

import json
import sys
from pathlib import Path

REQUIRED_KEYS = ["question_id", "dimension", "question_text"]
OPTIONAL_KEYS = [
    "version", "sub_dimension", "negative_prompt", "risk_tags",
    "expected_behavior", "language", "reference_answer", "metadata", "release_date",
]


def validate_question(q: dict, index: int) -> list:
    errors = []
    for key in REQUIRED_KEYS:
        if key not in q:
            errors.append(f"题目 #{index}: 缺少必填字段 '{key}'")
        elif key == "dimension" and q.get(key) != "text_to_image":
            errors.append(f"题目 #{index}: dimension 应为 'text_to_image'，当前为 '{q.get(key)}'")
    if not isinstance(q.get("question_text"), str) or not (q.get("question_text") or "").strip():
        errors.append(f"题目 #{index}: question_text 必须为非空字符串")
    return errors


def validate_file(file_path: Path) -> list:
    errors = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"{file_path}: JSON 解析失败 - {e}"]
    except Exception as e:
        return [f"{file_path}: 读取失败 - {e}"]
    if not isinstance(data, list):
        return [f"{file_path}: 根节点应为数组"]
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"{file_path} 题目 #{i}: 应为对象")
            continue
        errors.extend(validate_question(item, i))
    return errors


def main():
    base = Path(__file__).resolve().parent.parent
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if not target.is_absolute():
            target = base / target
    else:
        target = base / "livesecbench" / "question_set" / "text_to_image"
    if not target.exists():
        print(f"路径不存在: {target}")
        sys.exit(1)
    all_errors = []
    if target.is_file():
        all_errors = validate_file(target)
    else:
        for p in target.glob("*.json"):
            all_errors.extend(validate_file(p))
    if all_errors:
        for e in all_errors:
            print(e)
        sys.exit(1)
    print("校验通过")
    sys.exit(0)


if __name__ == "__main__":
    main()
