"""交接契约：每个环节的输入输出都有明确的形状，不合格不许往下一步走。

为什么值得为它写一百多行校验器：多智能体最大的一类失败是「规格问题」——
链条越长，模糊的规格被反复解释，误差在每次交接处放大。这里做的就是把
「请你按要求输出」从提示词里的**请求**，变成代码里的**约束**。

本模块只依赖标准库，不引入 jsonschema / pydantic。
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

from arksol.errors import ContractError

# --------------------------------------------------------------------------------------
# 最小 schema 校验器
#
# 支持的关键字（够用即可，不追求 JSON Schema 全集）：
#   type / enum / required / properties / additionalProperties
#   items / minItems / maxItems
#   minLength / maxLength / pattern
#   minimum / maximum
# --------------------------------------------------------------------------------------

_TYPE_NAMES = {
    "object": "对象",
    "array": "数组",
    "string": "字符串",
    "integer": "整数",
    "number": "数字",
    "boolean": "布尔值",
}


def _json_type_of(value: Any) -> str:
    """返回 JSON 视角下的类型名（注意 bool 在 Python 里是 int 的子类）。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    return type(value).__name__


def _matches_type(value: Any, expected: str) -> bool:
    actual = _json_type_of(value)
    if expected == "number":
        return actual in ("number", "integer")
    return actual == expected


def validate(instance: Any, schema: Mapping[str, Any], *, schema_name: str = "<inline>", path: str = "$") -> None:
    """校验 instance 是否符合 schema。不符合则抛 ContractError（带出错路径）。"""
    expected = schema.get("type")
    if expected is not None:
        if not _matches_type(instance, expected):
            raise ContractError(
                schema_name,
                path,
                f"期望 {_TYPE_NAMES.get(expected, expected)}，实际是 {_json_type_of(instance)}",
            )

    if "enum" in schema and instance not in schema["enum"]:
        raise ContractError(schema_name, path, f"取值必须是 {schema['enum']} 之一，实际是 {instance!r}")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            raise ContractError(schema_name, path, f"长度至少 {schema['minLength']}，实际 {len(instance)}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            raise ContractError(schema_name, path, f"长度最多 {schema['maxLength']}，实际 {len(instance)}")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            raise ContractError(schema_name, path, f"不匹配正则 {schema['pattern']!r}：{instance!r}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise ContractError(schema_name, path, f"不能小于 {schema['minimum']}，实际 {instance}")
        if "maximum" in schema and instance > schema["maximum"]:
            raise ContractError(schema_name, path, f"不能大于 {schema['maximum']}，实际 {instance}")

    if isinstance(instance, Mapping):
        for key in schema.get("required", []):
            if key not in instance:
                raise ContractError(schema_name, path, f"缺少必填字段 {key!r}")

        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = [k for k in instance if k not in props]
            if extra:
                raise ContractError(schema_name, path, f"出现未定义的字段 {extra}")

        for key, sub_schema in props.items():
            if key in instance:
                validate(instance[key], sub_schema, schema_name=schema_name, path=f"{path}.{key}")

    if isinstance(instance, (list, tuple)):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            raise ContractError(schema_name, path, f"至少 {schema['minItems']} 项，实际 {len(instance)}")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise ContractError(schema_name, path, f"最多 {schema['maxItems']} 项，实际 {len(instance)}")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(instance):
                validate(item, item_schema, schema_name=schema_name, path=f"{path}[{index}]")


# --------------------------------------------------------------------------------------
# 契约定义
# --------------------------------------------------------------------------------------

CONSTRAINT_KINDS = ["cost", "latency", "compliance", "concurrency", "quality", "other"]
CONFIDENCE_LEVELS = ["高", "中", "低"]

EVIDENCE_ITEM = {
    "type": "object",
    "required": ["id", "topic", "text", "source_id", "confidence"],
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "pattern": r"^EV-[A-Za-z0-9\-]+$"},
        "topic": {"type": "string", "minLength": 1},
        "text": {"type": "string", "minLength": 4},
        "source_id": {"type": "string", "pattern": r"^S\d+$"},
        "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
    },
}

CONSTRAINT = {
    "type": "object",
    "required": ["kind", "detail", "critical"],
    "additionalProperties": False,
    "properties": {
        "kind": {"type": "string", "enum": CONSTRAINT_KINDS},
        "detail": {"type": "string", "minLength": 2},
        "critical": {"type": "boolean"},
    },
}

# 追问必须说明「为什么这条会改变方案」—— 说不出会改变什么的问题不该问
QUESTION = {
    "type": "object",
    "required": ["question", "why_critical"],
    "additionalProperties": False,
    "properties": {
        "question": {"type": "string", "minLength": 4},
        "why_critical": {"type": "string", "minLength": 4},
    },
}

CLAIM = {
    "type": "object",
    "required": ["text", "evidence_ref", "kind"],
    "additionalProperties": False,
    "properties": {
        # evidence_ref 强制以 EV- 开头：**没有引用的断言在契约层就不合法**
        "text": {"type": "string", "minLength": 4},
        "evidence_ref": {"type": "string", "pattern": r"^EV-"},
        "kind": {"type": "string", "enum": ["capability", "limit", "price", "compliance", "performance"]},
    },
}

SCHEMAS: dict[str, dict] = {
    # Intake 的产出
    "Requirement": {
        "type": "object",
        "required": ["business_scenario", "capabilities", "constraints", "open_questions", "sufficient"],
        "additionalProperties": False,
        "properties": {
            "business_scenario": {"type": "string", "minLength": 2},
            "capabilities": {"type": "array", "items": {"type": "string", "minLength": 2}, "minItems": 1},
            "constraints": {"type": "array", "items": CONSTRAINT},
            "open_questions": {"type": "array", "items": QUESTION},
            # sufficient=False 表示「信息不足」—— 此时编排器输出追问清单并**结束**，不出方案
            "sufficient": {"type": "boolean"},
        },
    },
    # 追问清单：由编排器从 Requirement.open_questions 组装，不另外调用模型
    "FollowUp": {
        "type": "object",
        "required": ["questions", "assumptions"],
        "additionalProperties": False,
        "properties": {
            "questions": {"type": "array", "minItems": 1, "items": QUESTION},
            "assumptions": {"type": "array", "items": {"type": "string", "minLength": 2}},
        },
    },
    # 取证环节的产出
    "EvidenceBundle": {
        "type": "object",
        "required": ["topic", "items", "missing"],
        "additionalProperties": False,
        "properties": {
            "topic": {"type": "string", "minLength": 2},
            "items": {"type": "array", "items": EVIDENCE_ITEM},
            "missing": {"type": "array", "items": {"type": "string", "minLength": 2}},
        },
    },
    # Writer 的产出：每条断言必须挂一条证据
    #
    # evidence_ref 的约定：
    #   EV-xxx  → 指向本次运行证据表里的一条证据
    #   EV-NONE → 显式的「我凭记忆写的，没有证据」标记（基线 A 会大量使用）
    # 有了 EV-NONE，审计器就能把「没有证据支撑的断言」和「没标来源的断言」分开统计。
    "SolutionDraft": {
        "type": "object",
        "required": ["summary", "model_choice", "deployment", "claims", "followup_questions"],
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string", "minLength": 10},
            "model_choice": {"type": "string", "minLength": 2},
            "deployment": {"type": "string", "minLength": 2},
            "claims": {"type": "array", "items": CLAIM, "minItems": 1},
            # 三条流水线都要显式回答「我还需要问什么」—— 空数组表示「不需要追问」。
            # 这样「该追问时有没有追问」这条指标对三条流水线是同一把尺子。
            "followup_questions": {"type": "array", "items": {"type": "string", "minLength": 2}},
        },
    },
    # Critic 的产出：只认结构化结论，不认散文式评价
    "Critique": {
        "type": "object",
        "required": ["verdict", "issues", "checked_claims"],
        "additionalProperties": False,
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "revise"]},
            "checked_claims": {"type": "integer", "minimum": 0},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["type", "detail"],
                    "additionalProperties": False,
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["unsupported_claim", "uncovered_constraint", "other"],
                        },
                        "detail": {"type": "string", "minLength": 2},
                    },
                },
            },
        },
    },
}


def validate_named(instance: Any, schema_name: str) -> None:
    """按名字取契约并校验。名字不存在时直接抛 KeyError（这是代码 bug，不是数据问题）。"""
    validate(instance, SCHEMAS[schema_name], schema_name=schema_name)


# --------------------------------------------------------------------------------------
# 从模型自由文本里抠出符合契约的对象
# --------------------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """从可能夹带废话/围栏的模型输出里抠出 JSON 对象。

    支持的三种形态：纯 JSON、```json 围栏、前后有解释性文字。
    局限：用「第一个 { 到最后一个 }」做兜底，遇到字符串里含大括号的极端情况会失败 ——
    这种情况下让模型重试，比在这里写一个容错解析器更划算。
    """
    if not isinstance(text, str) or not text.strip():
        raise ContractError("<json>", "$", "输出为空")

    candidates = [text.strip()]
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.insert(0, fence.group(1).strip())

    stripped = text.strip()
    start = min([i for i in (stripped.find("{"), stripped.find("[")) if i >= 0], default=-1)
    if start >= 0:
        end = max(stripped.rfind("}"), stripped.rfind("]"))
        if end > start:
            candidates.append(stripped[start : end + 1])

    last_error: Optional[str] = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
    raise ContractError("<json>", "$", f"无法解析为 JSON（{last_error}）")


def parse_contract(text: str, schema_name: str) -> Any:
    """抠 JSON + 校验契约。任何一步失败都抛 ContractError —— 调用方据此决定本地重试。"""
    data = extract_json(text)
    validate_named(data, schema_name)
    return data
