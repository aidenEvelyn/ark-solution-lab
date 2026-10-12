"""交接契约的测试。

重点不是「校验器能跑」，而是**它拦不拦得住** —— 所以负面用例比正面用例多。
"""

from __future__ import annotations

import unittest

from arksol.contracts import SCHEMAS, extract_json, parse_contract, validate, validate_named
from arksol.errors import ContractError


def good_requirement() -> dict:
    return {
        "business_scenario": "企业内部知识库问答",
        "capabilities": ["文档检索", "多轮问答"],
        "constraints": [
            {"kind": "compliance", "detail": "数据不能出内网", "critical": True},
            {"kind": "latency", "detail": "首字延迟 2 秒内", "critical": False},
        ],
        "open_questions": [{"question": "并发量峰值是多少", "why_critical": "决定要不要专属部署"}],
        "sufficient": True,
    }


def good_draft(evidence_ref: str = "EV-001") -> dict:
    return {
        "summary": "建议用长上下文模型直接吃下全部文档，避免引入向量库。",
        "model_choice": "doubao-seed-evolving",
        "deployment": "公有云在线推理",
        "claims": [
            {"text": "该模型上下文为 1024k 量级", "evidence_ref": evidence_ref, "kind": "limit"},
            {"text": "该模型限流约 RPM 500", "evidence_ref": evidence_ref, "kind": "performance"},
        ],
        "followup_questions": [],
    }


class TestSchemaValidation(unittest.TestCase):
    def test_accepts_good_requirement(self) -> None:
        validate_named(good_requirement(), "Requirement")

    def test_accepts_good_draft(self) -> None:
        validate_named(good_draft(), "SolutionDraft")

    def test_rejects_missing_required_field(self) -> None:
        payload = good_requirement()
        del payload["open_questions"]
        with self.assertRaises(ContractError) as ctx:
            validate_named(payload, "Requirement")
        self.assertIn("open_questions", str(ctx.exception))

    def test_rejects_wrong_enum_value(self) -> None:
        payload = good_requirement()
        payload["constraints"][0]["kind"] = "预算"  # 必须是英文枚举
        with self.assertRaises(ContractError) as ctx:
            validate_named(payload, "Requirement")
        self.assertIn("$.constraints[0].kind", str(ctx.exception))

    def test_rejects_undefined_extra_field(self) -> None:
        payload = good_requirement()
        payload["mood"] = "开心"
        with self.assertRaises(ContractError) as ctx:
            validate_named(payload, "Requirement")
        self.assertIn("未定义的字段", str(ctx.exception))

    def test_rejects_boolean_where_integer_expected(self) -> None:
        # Python 里 bool 是 int 的子类，必须显式挡住
        with self.assertRaises(ContractError):
            validate(True, {"type": "integer"}, schema_name="T")

    def test_rejects_short_string(self) -> None:
        payload = good_requirement()
        payload["business_scenario"] = "x"
        with self.assertRaises(ContractError):
            validate_named(payload, "Requirement")

    def test_rejects_empty_capability_list(self) -> None:
        payload = good_requirement()
        payload["capabilities"] = []
        with self.assertRaises(ContractError):
            validate_named(payload, "Requirement")

    def test_error_message_carries_path(self) -> None:
        with self.assertRaises(ContractError) as ctx:
            validate({"a": [{"b": 1}]}, {"properties": {"a": {"items": {"properties": {"b": {"type": "string"}}}}}}, schema_name="T")
        self.assertIn("$.a[0].b", str(ctx.exception))


class TestClaimMustCiteEvidence(unittest.TestCase):
    """本项目最核心的一条不变量：没有引用的断言，在契约层就不合法。"""

    def test_rejects_claim_without_evidence_ref(self) -> None:
        draft = good_draft()
        draft["claims"][0]["evidence_ref"] = "E-1"  # 少了 EV- 前缀
        with self.assertRaises(ContractError) as ctx:
            validate_named(draft, "SolutionDraft")
        self.assertIn("$.claims[0].evidence_ref", str(ctx.exception))

    def test_accepts_explicit_no_evidence_marker(self) -> None:
        # EV-NONE 是「我凭记忆写的，没有证据」的显式标记 —— 合法，但会被审计器计入无依据断言
        validate_named(good_draft(evidence_ref="EV-NONE"), "SolutionDraft")

    def test_rejects_draft_with_zero_claims(self) -> None:
        draft = good_draft()
        draft["claims"] = []
        with self.assertRaises(ContractError):
            validate_named(draft, "SolutionDraft")

    def test_requires_followup_questions_field(self) -> None:
        # 三条流水线都必须显式回答「我还需要问什么」，空数组也算回答
        draft = good_draft()
        del draft["followup_questions"]
        with self.assertRaises(ContractError) as ctx:
            validate_named(draft, "SolutionDraft")
        self.assertIn("followup_questions", str(ctx.exception))

    def test_rejects_bad_source_id_format(self) -> None:
        bundle = {
            "topic": "模型选型",
            "items": [{"id": "EV-1", "text": "上下文 1024k", "source_id": "SRC1", "confidence": "中"}],
            "missing": [],
        }
        with self.assertRaises(ContractError) as ctx:
            validate_named(bundle, "EvidenceBundle")
        self.assertIn("source_id", str(ctx.exception))

    def test_accepts_well_formed_bundle(self) -> None:
        bundle = {
            "topic": "模型选型",
            "items": [{"id": "EV-1", "text": "上下文 1024k", "source_id": "S01", "confidence": "中"}],
            "missing": ["未查到最新版本单价"],
        }
        validate_named(bundle, "EvidenceBundle")

    def test_rejects_redundant_topic_on_evidence_item(self) -> None:
        # 主题已经在 bundle 上；要求每条证据重复填一遍，只会制造无谓的契约失败
        bundle = {
            "topic": "模型选型",
            "items": [
                {"id": "EV-1", "topic": "模型选型", "text": "上下文 1024k", "source_id": "S01", "confidence": "中"}
            ],
            "missing": [],
        }
        with self.assertRaises(ContractError):
            validate_named(bundle, "EvidenceBundle")

    def test_rejects_question_without_why_critical(self) -> None:
        # 追问必须说明「这条会改变什么」—— 说不出改变什么的问题不该问
        payload = good_requirement()
        payload["open_questions"] = [{"question": "你们的预算是多少呢"}]
        with self.assertRaises(ContractError) as ctx:
            validate_named(payload, "Requirement")
        self.assertIn("why_critical", str(ctx.exception))


class TestExtractJson(unittest.TestCase):
    def test_plain_json(self) -> None:
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self) -> None:
        text = '好的，这是结果：\n```json\n{"a": 1, "b": [2, 3]}\n```\n以上。'
        self.assertEqual(extract_json(text), {"a": 1, "b": [2, 3]})

    def test_fence_without_language(self) -> None:
        self.assertEqual(extract_json('```\n{"a": 1}\n```'), {"a": 1})

    def test_prose_around_object(self) -> None:
        self.assertEqual(extract_json('先说明一下。{"a": [1, 2]} 后面还有话。'), {"a": [1, 2]})

    def test_empty_raises(self) -> None:
        with self.assertRaises(ContractError):
            extract_json("   ")

    def test_garbage_raises(self) -> None:
        with self.assertRaises(ContractError):
            extract_json("这里完全没有 JSON")


class TestParseContract(unittest.TestCase):
    def test_parses_valid_requirement_from_messy_text(self) -> None:
        import json

        text = "我先梳理一下：\n```json\n" + json.dumps(good_requirement(), ensure_ascii=False) + "\n```"
        parsed = parse_contract(text, "Requirement")
        self.assertEqual(parsed["constraints"][0]["kind"], "compliance")

    def test_bad_json_raises(self) -> None:
        with self.assertRaises(ContractError):
            parse_contract("{不是 JSON}", "Requirement")

    def test_valid_json_but_wrong_schema_raises(self) -> None:
        with self.assertRaises(ContractError):
            parse_contract('{"summary": "太短"}', "SolutionDraft")


class TestSchemaRegistry(unittest.TestCase):
    def test_all_expected_schemas_present(self) -> None:
        for name in ("Requirement", "FollowUp", "EvidenceBundle", "SolutionDraft", "Critique"):
            self.assertIn(name, SCHEMAS)

    def test_unknown_schema_name_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            validate_named({}, "Nope")


if __name__ == "__main__":
    unittest.main()
