"""流水线端到端测试。

这里最重要的一组用例是**故障注入下的对照**：
同一个坏输出，在「有审校」和「没审校」两条流水线上，结局必须不一样 ——
否则「Critic 有用」这句话就没有证据。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from arksol.errors import BudgetExceeded
from arksol.llm import Fault, MockProvider
from arksol.pipeline import (
    PIPELINE_NAMES,
    Harness,
    compute_metrics,
    entry_source,
    gather_evidence,
    normalize_evidence,
    run_evidence_once,
    run_multi,
    run_no_evidence,
    run_pipeline,
    snippets_to_evidence,
)
from arksol.budget import Budget, Meter
from arksol.state import RunState
from arksol.verify import NO_EVIDENCE_MARK

GOOD_INPUT = "我们想做个内部知识库问答，给客服团队用，但数据不能出内网，QPS 大概 50，预算每月两千左右。"
VAGUE_INPUT = "想做个 AI 的东西"


def harness_with(provider=None, **kwargs) -> Harness:
    return Harness.build(provider or MockProvider(), **kwargs)


class TestHarnessRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = harness_with()

    def test_snippet_carries_source_and_confidence(self) -> None:
        snippets = self.harness.retrieve("数据合规 内网 私有化", top_k=3)
        self.assertTrue(snippets)
        for snippet in snippets:
            self.assertTrue(snippet["entry_id"])
            self.assertTrue(snippet["source_id"])
            self.assertIn(snippet["confidence"], ("高", "中", "低"))

    def test_entry_source_takes_worst_confidence(self) -> None:
        entry = self.harness.kb.get("model.doubao-seed-2-1-turbo")  # 来源 S01(中) + S08(低)
        _, confidence = entry_source(self.harness.kb, entry)
        self.assertEqual(confidence, "低")

    def test_repeated_retrieval_trips_repeat_guard(self) -> None:
        for _ in range(2):
            self.harness.retrieve("鉴权 404", top_k=1)
        with self.assertRaises(Exception) as ctx:
            self.harness.retrieve("鉴权 404", top_k=1)
        self.assertIn("重复动作熔断", str(ctx.exception))


class TestEvidenceNormalization(unittest.TestCase):
    def test_ids_are_renumbered_across_bundles(self) -> None:
        bundles = [
            {"topic": "a", "items": [{"id": "EV-001", "text": "x1"}], "missing": []},
            {"topic": "b", "items": [{"id": "EV-001", "text": "x2"}, {"id": "EV-002", "text": "x3"}], "missing": []},
        ]
        normalized, flat = normalize_evidence(bundles)
        self.assertEqual([item["id"] for item in flat], ["EV-001", "EV-002", "EV-003"])
        self.assertEqual([item["id"] for item in normalized[1]["items"]], ["EV-002", "EV-003"])

    def test_snippets_to_evidence_numbers_from_one(self) -> None:
        items = snippets_to_evidence([{"text": "正文", "source_id": "S01", "confidence": "中"}])
        self.assertEqual(items[0]["id"], "EV-001")
        self.assertEqual(items[0]["source_id"], "S01")


class TestComputeMetrics(unittest.TestCase):
    def test_without_draft(self) -> None:
        metrics = compute_metrics(draft=None, requirement=None, evidence_ids=set())
        self.assertEqual(metrics["claims_total"], 0)
        self.assertFalse(metrics["followup_asked"])

    def test_followup_counts_as_asked(self) -> None:
        metrics = compute_metrics(
            draft=None,
            requirement=None,
            evidence_ids=set(),
            followup={"questions": [{"question": "x", "why_critical": "y"}], "assumptions": []},
        )
        self.assertTrue(metrics["followup_asked"])


class TestNoEvidencePipeline(unittest.TestCase):
    def test_runs_and_produces_unbacked_claims(self) -> None:
        provider = MockProvider()
        record = run_no_evidence(GOOD_INPUT, harness=harness_with(provider))
        self.assertTrue(record.ok)
        self.assertIsNotNone(record.draft)
        self.assertEqual(record.evidence_count, 0)
        self.assertEqual(record.metrics["claims_unsupported"], record.metrics["claims_total"])
        # 老实标 EV-NONE，而不是编一个假的来源 —— 两者被分开统计
        self.assertEqual(record.metrics["claims_dangling"], 0)

    def test_does_not_touch_the_knowledge_base(self) -> None:
        provider = MockProvider()
        run_no_evidence(GOOD_INPUT, harness=harness_with(provider))
        self.assertNotIn("evidence", provider.call_counts)


class TestEvidenceOncePipeline(unittest.TestCase):
    def test_claims_are_backed_by_evidence(self) -> None:
        record = run_evidence_once(GOOD_INPUT, harness=harness_with())
        self.assertTrue(record.ok)
        self.assertGreater(record.evidence_count, 0)
        self.assertEqual(record.metrics["claims_unsupported"], 0)
        self.assertEqual(record.metrics["coverage_rate"], 1.0)

    def test_no_critic_runs(self) -> None:
        provider = MockProvider()
        record = run_evidence_once(GOOD_INPUT, harness=harness_with(provider))
        self.assertIsNone(record.critique)
        self.assertEqual(record.rounds, 0)
        self.assertNotIn("critic", provider.call_counts)


class TestMultiPipeline(unittest.TestCase):
    def test_runs_three_parallel_evidence_agents(self) -> None:
        provider = MockProvider()
        record = run_multi(GOOD_INPUT, harness=harness_with(provider))
        self.assertTrue(record.ok)
        self.assertEqual(provider.call_counts["evidence"], 3)
        self.assertGreaterEqual(record.evidence_count, 3)

    def test_critic_passes_on_clean_run(self) -> None:
        record = run_multi(GOOD_INPUT, harness=harness_with())
        self.assertIsNotNone(record.critique)
        self.assertEqual(record.critique["verdict"], "pass")
        self.assertEqual(record.rounds, 1)

    def test_writer_is_single_threaded(self) -> None:
        # 写入只在 Writer 里发生：三条取证 agent 各自产出的是证据，不是方案片段
        provider = MockProvider()
        record = run_multi(GOOD_INPUT, harness=harness_with(provider))
        self.assertEqual(provider.call_counts["writer"], record.rounds)
        self.assertIsNotNone(record.draft)


class TestFollowupPath(unittest.TestCase):
    def test_vague_input_stops_before_writing_a_plan(self) -> None:
        provider = MockProvider()
        record = run_multi(VAGUE_INPUT, harness=harness_with(provider))
        self.assertTrue(record.ok)
        self.assertIsNotNone(record.followup)
        self.assertIsNone(record.draft)
        self.assertNotIn("writer", provider.call_counts)

    def test_followup_questions_explain_why(self) -> None:
        record = run_multi(VAGUE_INPUT, harness=harness_with())
        for question in record.followup["questions"]:
            self.assertTrue(question["why_critical"])


class TestFaultInjectionAcrossPipelines(unittest.TestCase):
    """同一份坏输出，两种流水线两种结局 —— 这是「审校有用」的直接证据。"""

    def test_contract_error_is_absorbed_by_local_retry(self) -> None:
        provider = MockProvider(faults=[Fault(kind="malformed_json", tag="writer", at_call=1)])
        record = run_evidence_once(GOOD_INPUT, harness=harness_with(provider))
        self.assertTrue(record.ok)
        self.assertEqual(provider.call_counts["writer"], 2)  # 第一次坏，重试成功

    def test_fabricated_citation_survives_without_a_critic(self) -> None:
        provider = MockProvider(faults=[Fault(kind="unsupported_claim", tag="writer", at_call=1)])
        record = run_evidence_once(GOOD_INPUT, harness=harness_with(provider))
        self.assertTrue(record.ok)
        self.assertEqual(record.metrics["claims_dangling"], 1)  # 伪造的引用就这么发出去了

    def test_fabricated_citation_is_caught_and_rewritten_with_a_critic(self) -> None:
        provider = MockProvider(faults=[Fault(kind="unsupported_claim", tag="writer", at_call=1)])
        record = run_multi(GOOD_INPUT, harness=harness_with(provider))
        self.assertTrue(record.ok)
        self.assertEqual(record.rounds, 2)  # 退回一次后通过
        self.assertEqual(record.critique["verdict"], "pass")
        self.assertEqual(record.metrics["claims_dangling"], 0)

    def test_critic_gives_up_after_max_rewrites(self) -> None:
        provider = MockProvider(faults=[Fault(kind="unsupported_claim", tag="writer", at_call=1, times=9)])
        record = run_multi(GOOD_INPUT, harness=harness_with(provider))
        self.assertEqual(record.rounds, 3)  # 初次 + 2 次改写
        self.assertEqual(record.critique["verdict"], "revise")
        self.assertIn("未通过", record.stop_reason or "")

    def test_constraint_ignored_is_caught_by_rules(self) -> None:
        provider = MockProvider(faults=[Fault(kind="constraint_ignored", tag="writer", at_call=1, times=9)])
        record = run_multi(GOOD_INPUT, harness=harness_with(provider))
        self.assertEqual(record.critique["verdict"], "revise")
        types = {issue["type"] for issue in record.critique["issues"]}
        self.assertIn("uncovered_constraint", types)

    def test_fault_only_affects_targeted_pipeline_stage(self) -> None:
        provider = MockProvider(faults=[Fault(kind="malformed_json", tag="critic")])
        record = run_multi(GOOD_INPUT, harness=harness_with(provider))
        self.assertTrue(record.ok)  # 规则层直接判过，根本没走到模型那次调用


class TestBudgetAndTrace(unittest.TestCase):
    def test_budget_stop_is_recorded_not_swallowed(self) -> None:
        harness = harness_with(budget=Budget(max_tokens=1))
        record = run_multi(GOOD_INPUT, harness=harness)
        self.assertFalse(record.ok)
        self.assertIn("BudgetExceeded", record.error or "")

    def test_trace_file_is_written(self) -> None:
        from arksol.trace import Tracer, read_trace

        with tempfile.TemporaryDirectory() as tmp:
            tracer = Tracer(Path(tmp) / "run.jsonl")
            harness = harness_with(tracer=tracer)
            try:
                record = run_multi(GOOD_INPUT, harness=harness)
            finally:
                tracer.close()
            self.assertTrue(record.ok)
            records = read_trace(tracer.path)
        self.assertTrue(records)
        agents = {record["agent"] for record in records}
        self.assertIn("intake", agents)
        self.assertIn("writer", agents)
        self.assertTrue(any(agent.startswith("evidence:") for agent in agents))


class TestPipelineRegistry(unittest.TestCase):
    def test_known_names(self) -> None:
        self.assertEqual(PIPELINE_NAMES, ("no-evidence", "evidence-once", "multi"))

    def test_run_pipeline_dispatch(self) -> None:
        for name in PIPELINE_NAMES:
            record = run_pipeline(name, GOOD_INPUT, harness=harness_with())
            self.assertEqual(record.pipeline, name)
            self.assertTrue(record.ok)

    def test_unknown_pipeline_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_pipeline("我编的流水线", GOOD_INPUT, harness=harness_with())


class TestRecordSerialization(unittest.TestCase):
    def test_to_dict_is_json_serializable(self) -> None:
        import json

        record = run_multi(GOOD_INPUT, harness=harness_with(), case_id="c1")
        payload = json.dumps(record.to_dict(), ensure_ascii=False)
        self.assertIn("c1", payload)


class TestUnusedImportsGuard(unittest.TestCase):
    """提醒：这段测试本身也是文档 —— 改动流水线时先把上面这套跑绿。"""

    def test_no_evidence_marker_is_the_documented_constant(self) -> None:
        self.assertEqual(NO_EVIDENCE_MARK, "EV-NONE")


if __name__ == "__main__":
    unittest.main()
