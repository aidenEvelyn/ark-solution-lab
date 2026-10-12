"""LLM 抽象层测试。

两条主线：
1. mock 的每一条产出都必须**符合契约** —— 否则管线测试就没有意义；
2. 故障注入必须**真的能造出坏输出** —— 否则「契约层拦得住」这句话没有证据。
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from arksol.contracts import parse_contract
from arksol.errors import ContractError, ProviderError
from arksol.llm import FAULT_KINDS, Fault, MockProvider, OpenAICompatProvider, get_provider
from arksol.llm.base import LLMRequest, render_user_prompt
from arksol.llm.openai_compat import DEFAULT_BASE_URL
from arksol.verify import check_constraints, classify_claims

SNIPPETS = [
    {
        "entry_id": "model.doubao-seed-evolving",
        "source_id": "S01",
        "confidence": "中",
        "text": "Doubao-Seed-Evolving：上下文 1024k，最大回答 256k，限流约 RPM 500 / TPM 1M。",
    },
    {
        "entry_id": "guide.pricing-model",
        "source_id": "S05",
        "confidence": "高",
        "text": "按 token 后付费；多数豆包模型按输入长度分段计价，缓存命中另有单价。",
    },
]


def intake_request(text: str) -> LLMRequest:
    return LLMRequest(system="你是售前需求分析师。", user=text, tag="intake", schema_name="Requirement", context={"user_input": text})


def evidence_request(snippets=SNIPPETS, topic="模型选型") -> LLMRequest:
    return LLMRequest(
        system="你是取证工程师。",
        user="请整理证据。",
        tag="evidence",
        schema_name="EvidenceBundle",
        context={"topic": topic, "snippets": list(snippets)},
    )


def requirement(sufficient: bool = True) -> dict:
    return {
        "business_scenario": "内部知识库问答",
        "capabilities": ["知识库问答"],
        "constraints": [{"kind": "compliance", "detail": "数据不能出内网", "critical": True}],
        "open_questions": [],
        "sufficient": sufficient,
    }


def writer_request(items) -> LLMRequest:
    return LLMRequest(
        system="你是方案撰写者。",
        user="请写方案。",
        tag="writer",
        schema_name="SolutionDraft",
        context={"requirement": requirement(), "evidence_items": list(items)},
    )


def evidence_items() -> list:
    return [
        {
            "id": "EV-001",
            "topic": "模型选型",
            "text": "Doubao-Seed-Evolving 上下文 1024k，限流约 RPM 500。",
            "source_id": "S01",
            "confidence": "中",
        },
        {
            "id": "EV-002",
            "topic": "计费",
            "text": "按 token 后付费，多数豆包模型按输入长度分段计价。",
            "source_id": "S05",
            "confidence": "高",
        },
    ]


class TestMockIntake(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = MockProvider()

    def test_well_specified_input_is_sufficient(self) -> None:
        text = "我们想做个内部知识库问答，但数据不能出内网，QPS 大概 50，预算每月两千。"
        parsed = parse_contract(self.provider.complete(intake_request(text)).text, "Requirement")
        self.assertTrue(parsed["sufficient"])
        kinds = {c["kind"] for c in parsed["constraints"]}
        self.assertIn("compliance", kinds)
        self.assertIn("cost", kinds)
        self.assertEqual(parsed["open_questions"], [])

    def test_vague_input_triggers_followup(self) -> None:
        parsed = parse_contract(self.provider.complete(intake_request("想做个 AI 的东西")).text, "Requirement")
        self.assertFalse(parsed["sufficient"])
        self.assertTrue(parsed["open_questions"])
        for question in parsed["open_questions"]:
            self.assertIn("why_critical", question)

    def test_deterministic(self) -> None:
        text = "我们想做个内部知识库问答，但数据不能出内网。"
        first = self.provider.complete(intake_request(text)).text
        second = self.provider.complete(intake_request(text)).text
        self.assertEqual(first, second)


class TestMockEvidence(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = MockProvider()

    def test_bundle_is_contract_valid(self) -> None:
        bundle = parse_contract(self.provider.complete(evidence_request()).text, "EvidenceBundle")
        self.assertEqual(len(bundle["items"]), 2)
        self.assertEqual([item["id"] for item in bundle["items"]], ["EV-001", "EV-002"])
        self.assertEqual(bundle["items"][0]["source_id"], "S01")
        self.assertEqual(bundle["missing"], [])

    def test_no_snippets_reports_missing(self) -> None:
        bundle = parse_contract(self.provider.complete(evidence_request(snippets=[], topic="冷门主题")).text, "EvidenceBundle")
        self.assertEqual(bundle["items"], [])
        self.assertTrue(bundle["missing"])


class TestMockWriter(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = MockProvider()

    def test_claims_reference_real_evidence(self) -> None:
        draft = parse_contract(self.provider.complete(writer_request(evidence_items())).text, "SolutionDraft")
        refs = {claim["evidence_ref"] for claim in draft["claims"]}
        self.assertTrue(refs.issubset({"EV-001", "EV-002"}))
        self.assertEqual(classify_claims(draft, {"EV-001", "EV-002"})["unsupported_count"], 0)

    def test_deployment_follows_compliance_constraint(self) -> None:
        draft = parse_contract(self.provider.complete(writer_request(evidence_items())).text, "SolutionDraft")
        self.assertIn("私有化", draft["deployment"])
        coverage = check_constraints(requirement(), draft)
        self.assertEqual(coverage["coverage_rate"], 1.0)

    def test_without_evidence_marks_claims_as_no_evidence(self) -> None:
        draft = parse_contract(self.provider.complete(writer_request([])).text, "SolutionDraft")
        self.assertTrue(all(claim["evidence_ref"] == "EV-NONE" for claim in draft["claims"]))


class TestMockSingleBaseline(unittest.TestCase):
    def test_all_claims_are_unbacked(self) -> None:
        request = LLMRequest(
            system="你是方案专家。",
            user="请给方案。",
            tag="single",
            schema_name="SolutionDraft",
            context={"requirement": requirement()},
        )
        draft = parse_contract(MockProvider().complete(request).text, "SolutionDraft")
        self.assertTrue(all(claim["evidence_ref"] == "EV-NONE" for claim in draft["claims"]))
        self.assertEqual(classify_claims(draft, set())["dangling"], [])


class TestMockCritic(unittest.TestCase):
    def test_critique_is_contract_valid(self) -> None:
        request = LLMRequest(
            system="你是审校。",
            user="请审校。",
            tag="critic",
            schema_name="Critique",
            context={"draft": {"claims": [1, 2, 3]}},
        )
        critique = parse_contract(MockProvider().complete(request).text, "Critique")
        self.assertEqual(critique["verdict"], "pass")
        self.assertEqual(critique["checked_claims"], 3)


class TestFaultInjection(unittest.TestCase):
    def test_all_advertised_kinds_exist(self) -> None:
        self.assertEqual(len(FAULT_KINDS), 6)

    def test_malformed_json_breaks_parsing(self) -> None:
        provider = MockProvider(faults=[Fault(kind="malformed_json", tag="writer")])
        response = provider.complete(writer_request(evidence_items()))
        self.assertEqual(response.fault, "malformed_json")
        with self.assertRaises(ContractError):
            parse_contract(response.text, "SolutionDraft")

    def test_empty_output_breaks_parsing(self) -> None:
        provider = MockProvider(faults=[Fault(kind="empty", tag="writer")])
        with self.assertRaises(ContractError):
            parse_contract(provider.complete(writer_request(evidence_items())).text, "SolutionDraft")

    def test_missing_field_is_caught_by_contract(self) -> None:
        provider = MockProvider(faults=[Fault(kind="missing_field", tag="writer")])
        with self.assertRaises(ContractError) as ctx:
            parse_contract(provider.complete(writer_request(evidence_items())).text, "SolutionDraft")
        self.assertIn("followup_questions", str(ctx.exception))

    def test_unsupported_claim_creates_dangling_reference(self) -> None:
        provider = MockProvider(faults=[Fault(kind="unsupported_claim", tag="writer")])
        draft = parse_contract(provider.complete(writer_request(evidence_items())).text, "SolutionDraft")
        result = classify_claims(draft, {"EV-001", "EV-002"})
        self.assertEqual(len(result["dangling"]), 1)
        self.assertIn("EV-999", result["dangling"][0]["evidence_ref"])

    def test_unknown_source_is_produced(self) -> None:
        provider = MockProvider(faults=[Fault(kind="unknown_source", tag="evidence")])
        bundle = parse_contract(provider.complete(evidence_request()).text, "EvidenceBundle")
        self.assertEqual(bundle["items"][0]["source_id"], "S99")

    def test_constraint_ignored_drops_coverage(self) -> None:
        provider = MockProvider(faults=[Fault(kind="constraint_ignored", tag="writer")])
        draft = parse_contract(provider.complete(writer_request(evidence_items())).text, "SolutionDraft")
        coverage = check_constraints(requirement(), draft)
        self.assertLess(coverage["coverage_rate"], 1.0)

    def test_fault_only_fires_at_targeted_call(self) -> None:
        provider = MockProvider(faults=[Fault(kind="malformed_json", tag="writer", at_call=2)])
        first = provider.complete(writer_request(evidence_items()))
        second = provider.complete(writer_request(evidence_items()))
        self.assertIsNone(first.fault)
        self.assertEqual(second.fault, "malformed_json")

    def test_fault_scoped_by_tag(self) -> None:
        provider = MockProvider(faults=[Fault(kind="malformed_json", tag="writer")])
        self.assertIsNone(provider.complete(intake_request("随便写点什么内容")).fault)

    def test_unknown_fault_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Fault(kind="让模型自己承认错误")

    def test_fault_accepts_plain_dict(self) -> None:
        provider = MockProvider(faults=[{"kind": "empty", "tag": "writer"}])
        self.assertEqual(provider.faults[0].kind, "empty")


class TestCallCounting(unittest.TestCase):
    def test_counts_per_tag(self) -> None:
        provider = MockProvider()
        provider.complete(intake_request("我们想做个内部知识库问答，数据不能出内网。"))
        provider.complete(writer_request(evidence_items()))
        provider.complete(writer_request(evidence_items()))
        self.assertEqual(provider.call_counts, {"intake": 1, "writer": 2})
        self.assertEqual(provider.total_calls, 3)


class TestProviderFactory(unittest.TestCase):
    def test_mock(self) -> None:
        self.assertIsInstance(get_provider("mock"), MockProvider)

    def test_unknown_name(self) -> None:
        with self.assertRaises(ProviderError):
            get_provider("某个我没实现的东西")

    def test_ark_reads_environment(self) -> None:
        with mock.patch.dict(
            os.environ, {"ARK_API_KEY": "fake-key-for-test", "ARK_MODEL": "doubao-test"}, clear=True
        ):
            provider = get_provider("ark")
        self.assertIsInstance(provider, OpenAICompatProvider)
        self.assertEqual(provider.model, "doubao-test")

    def test_missing_key_is_a_clear_error_not_a_crash(self) -> None:
        with self.assertRaises(ProviderError) as ctx:
            OpenAICompatProvider.from_env(env={})
        self.assertIn("API Key", str(ctx.exception))

    def test_key_is_kept_private(self) -> None:
        provider = OpenAICompatProvider(api_key="super-secret-value")
        # 密钥只能待在私有属性里；对象的 repr / str / 公开属性都不得泄露它
        self.assertNotIn("super-secret-value", repr(provider))
        self.assertNotIn("super-secret-value", str(provider))
        public = {k: v for k, v in vars(provider).items() if not k.startswith("_")}
        self.assertNotIn("super-secret-value", repr(public))
        self.assertIn("_api_key", vars(provider))


class TestOpenAICompatConfig(unittest.TestCase):
    def test_default_base_url_already_contains_api_version(self) -> None:
        self.assertTrue(DEFAULT_BASE_URL.endswith("/api/v3"))
        self.assertFalse(DEFAULT_BASE_URL.endswith("/api/v3/v1"))

    def test_trailing_slash_is_trimmed(self) -> None:
        provider = OpenAICompatProvider(api_key="k", base_url="https://example.com/api/v3/")
        self.assertEqual(provider.base_url, "https://example.com/api/v3")

    def test_env_overrides(self) -> None:
        provider = OpenAICompatProvider.from_env(
            env={"ARK_API_KEY": "k", "ARK_BASE_URL": "https://x.test/v1", "ARK_MODEL": "m",
                 "ARK_PRICE_IN": "1.5", "ARK_PRICE_OUT": "6"}
        )
        self.assertEqual(provider.base_url, "https://x.test/v1")
        self.assertEqual(provider.model, "m")

    def test_render_user_prompt_embeds_json(self) -> None:
        text = render_user_prompt("请按契约输出。", {"a": "中文"})
        self.assertIn("请按契约输出。", text)
        self.assertIn("中文", text)


if __name__ == "__main__":
    unittest.main()
