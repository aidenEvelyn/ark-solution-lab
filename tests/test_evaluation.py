"""评测台测试。

重点保护两件事：
1. 用例文件坏了要**明确报错**，不能静默跳过 —— 否则评测结果会建立在少跑了几条用例的基础上；
2. 报告里那段「哪些数字有效」的声明**不能丢** —— 它是这个项目唯一可信的地方。
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from arksol.errors import ArksolError
from arksol.evaluation import (
    FAULT_SCENARIOS,
    load_cases,
    load_raw,
    rebuild_report,
    recompute_derived,
    render_report,
    run_fault_experiment,
    run_matrix,
    split_by_followup,
    write_csv,
    write_raw,
)
from arksol.evaluation.cases import default_cases_path
from arksol.llm import MockProvider

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "evaluation" / "cases.jsonl"


class TestCaseLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_cases(CASES_PATH)

    def test_loads_expected_count(self) -> None:
        self.assertEqual(len(self.cases), 20)

    def test_ids_unique(self) -> None:
        ids = [case.id for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_has_both_specified_and_underspecified(self) -> None:
        enough, need_ask = split_by_followup(self.cases)
        self.assertEqual(len(enough), 14)
        self.assertEqual(len(need_ask), 6)

    def test_every_case_has_note(self) -> None:
        for case in self.cases:
            self.assertTrue(case.note, f"{case.id} 没有写说明")

    def test_default_path_is_discoverable(self) -> None:
        found = default_cases_path()
        self.assertIsNotNone(found)
        self.assertTrue(found.exists())

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(ArksolError):
            load_cases(ROOT / "evaluation" / "不存在.jsonl")

    def test_broken_json_line_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text('{"id": "C1", "input": "正常"}\n{坏行}\n', encoding="utf-8")
            with self.assertRaises(ArksolError) as ctx:
                load_cases(path)
            self.assertIn("不是合法 JSON", str(ctx.exception))

    def test_duplicate_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(
                '{"id": "C1", "input": "第一条"}\n{"id": "C1", "input": "第二条"}\n', encoding="utf-8"
            )
            with self.assertRaises(ArksolError) as ctx:
                load_cases(path)
            self.assertIn("id 重复", str(ctx.exception))

    def test_missing_field_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text('{"id": "C1"}\n', encoding="utf-8")
            with self.assertRaises(ArksolError) as ctx:
                load_cases(path)
            self.assertIn("input", str(ctx.exception))


class TestRunMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = load_cases(CASES_PATH)[:2]  # 只取两条，保持测试快
        self._tmp = tempfile.TemporaryDirectory()
        self.trace_dir = Path(self._tmp.name) / "traces"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_matrix_shape(self) -> None:
        runs, summaries = run_matrix(
            self.cases, ["no-evidence", "evidence-once", "multi"], MockProvider, trace_dir=self.trace_dir
        )
        self.assertEqual(len(runs), 6)
        self.assertEqual([s.pipeline for s in summaries], ["no-evidence", "evidence-once", "multi"])
        for summary in summaries:
            self.assertEqual(summary.cases, 2)
            self.assertEqual(summary.ok, 2)

    def test_trace_file_written_per_pipeline(self) -> None:
        run_matrix(self.cases, ["multi"], MockProvider, trace_dir=self.trace_dir)
        trace_file = self.trace_dir / "multi.jsonl"
        self.assertTrue(trace_file.exists())
        lines = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        run_ids = {line["run_id"] for line in lines}
        self.assertEqual(run_ids, {case.id for case in self.cases})

    def test_no_evidence_pipeline_has_unbacked_claims(self) -> None:
        _runs, summaries = run_matrix(self.cases, ["no-evidence"], MockProvider, trace_dir=self.trace_dir)
        self.assertGreater(summaries[0].unsupported_rate, 0.9)

    def test_multi_costs_more_calls_than_evidence_once(self) -> None:
        _runs, summaries = run_matrix(
            self.cases, ["evidence-once", "multi"], MockProvider, trace_dir=self.trace_dir
        )
        by_name = {s.pipeline: s for s in summaries}
        self.assertGreater(by_name["multi"].avg_calls, by_name["evidence-once"].avg_calls)
        self.assertGreater(by_name["multi"].avg_tokens, by_name["evidence-once"].avg_tokens)

    def test_summary_serialization(self) -> None:
        _runs, summaries = run_matrix(self.cases, ["multi"], MockProvider, trace_dir=self.trace_dir)
        payload = summaries[0].to_dict()
        json.dumps(payload, ensure_ascii=False)
        for key in ("unsupported_rate", "coverage_rate", "avg_calls", "success_rate"):
            self.assertIn(key, payload)


class TestFaultExperiment(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = load_cases(CASES_PATH)[:2]
        self._tmp = tempfile.TemporaryDirectory()
        self.trace_dir = Path(self._tmp.name) / "traces"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_row_shape(self) -> None:
        rows = run_fault_experiment(
            self.cases,
            ["evidence-once", "multi"],
            trace_dir=self.trace_dir,
            limit=2,
            scenarios=FAULT_SCENARIOS[:1],
        )
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row.runs, 2)

    def test_critic_prevents_leakage_of_fabricated_citation(self) -> None:
        rows = run_fault_experiment(
            self.cases,
            ["evidence-once", "multi"],
            trace_dir=self.trace_dir,
            limit=2,
            scenarios=[FAULT_SCENARIOS[0]],  # 伪造引用（偶发）
        )
        by_pipeline = {row.pipeline: row for row in rows}
        self.assertEqual(by_pipeline["evidence-once"].leaked, 2)  # 没有审校：全漏出去
        self.assertEqual(by_pipeline["multi"].leaked, 0)  # 有审校：全部拦下

    def test_persistent_fault_cannot_be_fixed(self) -> None:
        rows = run_fault_experiment(
            self.cases,
            ["multi"],
            trace_dir=self.trace_dir,
            limit=1,
            scenarios=[FAULT_SCENARIOS[1]],  # 忽略约束（持续）
        )
        row = rows[0]
        self.assertEqual(row.caught, 1)  # 能发现
        self.assertEqual(row.leaked, 1)  # 但改不动 —— 审校不是万能的
        self.assertGreater(row.avg_rounds, 1)


class TestReportRendering(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cases = load_cases(CASES_PATH)[:2]
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp) / "traces"
            runs, summaries = run_matrix(
                cases, ["no-evidence", "evidence-once", "multi"], MockProvider, trace_dir=trace_dir
            )
            fault_rows = run_fault_experiment(
                cases, ["evidence-once", "multi"], trace_dir=trace_dir, limit=2, scenarios=FAULT_SCENARIOS[:1]
            )
        cls.runs = runs
        cls.summaries = summaries
        cls.fault_rows = fault_rows
        cls.meta = {
            "generated_at": "2026-10-12T00:00:00+08:00",
            "provider": "mock",
            "model": "mock-deterministic-v1",
            "cases": 2,
            "underspecified_cases": 0,
            "pipelines": ["no-evidence", "evidence-once", "multi"],
            "command": "python -m arksol eval --provider mock",
        }
        cls.report = render_report(summaries=summaries, fault_rows=fault_rows, meta=cls.meta)

    def test_has_honesty_section(self) -> None:
        self.assertIn("哪些数字有效", self.report)
        self.assertIn("不能证明", self.report)

    def test_mock_marks_quality_metrics_as_scripted(self) -> None:
        self.assertIn("反映脚本设定", self.report)
        self.assertIn("自证", self.report)

    def test_has_three_tables(self) -> None:
        self.assertIn("表 1", self.report)
        self.assertIn("表 2", self.report)
        self.assertIn("表 3", self.report)

    def test_conclusions_are_data_driven(self) -> None:
        self.assertIn("多智能体的开销是实打实的", self.report)
        self.assertIn("故障路径", self.report)

    def test_reproduce_commands_included(self) -> None:
        self.assertIn("python -m arksol eval --provider mock", self.report)
        self.assertIn("ARK_API_KEY", self.report)

    def test_real_provider_report_switches_wording(self) -> None:
        meta = dict(self.meta, provider="openai-compat", model="doubao-test")
        report = render_report(summaries=self.summaries, fault_rows=self.fault_rows, meta=meta)
        self.assertIn("doubao-test", report)
        self.assertNotIn("反映脚本设定", report)
        self.assertIn("不是账单", report)


class TestOutputWriters(unittest.TestCase):
    def test_csv_and_raw(self) -> None:
        cases = load_cases(CASES_PATH)[:2]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            runs, summaries = run_matrix(cases, ["multi"], MockProvider, trace_dir=out / "traces")
            write_csv(out / "results.csv", runs)
            write_raw(out / "raw.json", {"summaries": [s.to_dict() for s in summaries]})

            with (out / "results.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["pipeline"], "multi")
            self.assertIn("claims_unsupported", rows[0])

            payload = json.loads((out / "raw.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["summaries"]), 1)


class TestReportRebuild(unittest.TestCase):
    """报告必须是「可重算的视图」，而不是跑完就散掉的输出。

    真实模型的评测很贵，改一次措辞不该重跑全量。只要 raw.json 留着原始记录，
    报告就应该能随时重渲染，而且派生指标要能从逐条记录重算出来。
    """

    SUFFICIENT = "我们想做个内部知识库问答，给客服团队用，但数据不能出内网，QPS 大概 50，预算每月两千左右。"
    VAGUE = "想做个 AI 的东西"

    def _fixture(self, tmp: Path) -> Path:
        from arksol.evaluation.cases import Case

        cases = [Case(id="T1", input=self.SUFFICIENT), Case(id="T2", input=self.VAGUE, needs_followup=True)]
        runs, summaries = run_matrix(cases, ["no-evidence", "multi"], MockProvider, trace_dir=tmp / "traces")
        raw = tmp / "raw.json"
        write_raw(
            raw,
            {
                "meta": {
                    "generated_at": "2026-10-12T00:00:00+08:00",
                    "provider": "mock",
                    "model": "mock-deterministic-v1",
                    "cases": 2,
                    "pipelines": ["no-evidence", "multi"],
                    "cases_file": "evaluation/cases.jsonl",
                },
                "summaries": [s.to_dict() for s in summaries],
                "runs": [r.to_dict() for r in runs],
                "fault_rows": [],
            },
        )
        return raw

    def test_rebuild_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = self._fixture(Path(tmp))
            target = rebuild_report(raw)
            text = Path(target).read_text(encoding="utf-8")
        self.assertIn("# 对照实验结果", text)
        self.assertIn("no-evidence", text)

    def test_missing_derived_fields_are_recomputed_from_runs(self) -> None:
        # 模拟「raw.json 是更早版本生成的」：删掉后来才加的字段
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            raw = self._fixture(base)
            payload = json.loads(raw.read_text(encoding="utf-8"))
            for summary in payload["summaries"]:
                summary.pop("draft_followup_count", None)
                summary.pop("failed_cases", None)
            raw.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            _meta, summaries, runs, _faults = load_raw(raw)
            recompute_derived(summaries, runs)
            text = rebuild_report(raw)
            rendered = Path(text).read_text(encoding="utf-8")

        # 每条流水线各 2 条用例，其中 1 条该追问 → 重算后必须是 1/2，而不是默认的 0/2
        per_pipeline: dict[str, int] = {}
        for run in runs:
            key = run.pipeline
            per_pipeline[key] = per_pipeline.get(key, 0) + (
                1 if (run.metrics or {}).get("followup_asked") else 0
            )
        self.assertEqual(set(per_pipeline.values()), {1})
        self.assertIn("1/2", rendered)

    def test_load_raw_returns_typed_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = self._fixture(Path(tmp))
            meta, summaries, runs, faults = load_raw(raw)
        self.assertEqual(meta["provider"], "mock")
        self.assertEqual(len(summaries), 2)
        self.assertEqual(len(runs), 4)
        self.assertEqual(faults, [])
        self.assertTrue(all(hasattr(run, "case_id") for run in runs))

    def test_missing_raw_raises(self) -> None:
        with self.assertRaises(ArksolError):
            load_raw(Path("不存在的目录") / "raw.json")


class TestFaultScenarioCatalogue(unittest.TestCase):
    def test_scenarios_are_declared_with_description(self) -> None:
        self.assertGreaterEqual(len(FAULT_SCENARIOS), 3)
        for scenario in FAULT_SCENARIOS:
            self.assertTrue(scenario["kind"])
            self.assertTrue(scenario["label"])
            self.assertTrue(scenario["desc"])


if __name__ == "__main__":
    unittest.main()
