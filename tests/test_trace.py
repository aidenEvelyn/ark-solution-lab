"""trace 测试：能不能定位到「具体哪一步」，取决于 trace 本身没丢东西。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from arksol.trace import Tracer, read_trace, summarize


class TestTracerWritesJsonl(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "traces" / "run.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_creates_parent_dir_and_file(self) -> None:
        with Tracer(self.path) as tracer:
            tracer.emit(agent="intake", event="call")
        self.assertTrue(self.path.exists())

    def test_record_fields(self) -> None:
        with Tracer(self.path) as tracer:
            record = tracer.emit(
                agent="intake",
                event="call",
                payload={"user_input": "想做个知识库问答"},
                usage={"prompt_tokens": 10, "completion_tokens": 20},
                elapsed_ms=123.456,
            )
        self.assertEqual(record["agent"], "intake")
        self.assertEqual(record["level"], "info")
        self.assertIn("run_id", record)
        self.assertIn("ts", record)

        restored = read_trace(self.path)
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0]["usage"]["prompt_tokens"], 10)
        self.assertEqual(restored[0]["elapsed_ms"], 123.5)

    def test_multiple_emits_append(self) -> None:
        with Tracer(self.path) as tracer:
            tracer.emit(agent="a", event="e1")
            tracer.emit(agent="b", event="e2")
        self.assertEqual(len(read_trace(self.path)), 2)

    def test_chinese_survives_round_trip(self) -> None:
        with Tracer(self.path) as tracer:
            tracer.emit(agent="writer", event="call", payload={"note": "无依据断言：2 条"})
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("无依据断言", text)
        self.assertEqual(read_trace(self.path)[0]["payload"]["note"], "无依据断言：2 条")

    def test_line_is_valid_json(self) -> None:
        with Tracer(self.path) as tracer:
            tracer.emit(agent="a", event="e")
        line = self.path.read_text(encoding="utf-8").splitlines()[0]
        self.assertIsInstance(json.loads(line), dict)


class TestSpan(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "run.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_span_records_elapsed(self) -> None:
        with Tracer(self.path) as tracer:
            with tracer.span(agent="intake", event="call", payload={"step": 1}):
                pass
        records = read_trace(self.path)
        self.assertEqual(len(records), 1)
        self.assertIn("elapsed_ms", records[0])

    def test_span_marks_error_and_reraises(self) -> None:
        with Tracer(self.path) as tracer:
            with self.assertRaises(ValueError):
                with tracer.span(agent="writer", event="call"):
                    raise ValueError("契约校验失败")
        records = read_trace(self.path)
        self.assertEqual(records[0]["level"], "error")


class TestNullTracer(unittest.TestCase):
    def test_no_file_created_and_no_crash(self) -> None:
        tracer = Tracer(None)
        tracer.emit(agent="a", event="e", payload={"x": 1})
        tracer.close()
        self.assertIsNone(tracer.path)


class TestReadAndSummarize(unittest.TestCase):
    def test_tolerates_truncated_last_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            path.write_text('{"agent": "a", "event": "e", "level": "info"}\n{"agent": "b", "even', encoding="utf-8")
            records = read_trace(path)
            self.assertEqual(len(records), 1)

    def test_summarize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with Tracer(path) as tracer:
                tracer.emit(agent="intake", event="call", elapsed_ms=10)
                tracer.emit(agent="writer", event="call", elapsed_ms=20)
                tracer.emit(agent="critic", event="call", elapsed_ms=5, level="error")
            summary = summarize(read_trace(path))
        self.assertEqual(summary["records"], 3)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["by_agent"]["writer"], 1)
        self.assertAlmostEqual(summary["total_elapsed_ms"], 35.0)

    def test_summarize_empty(self) -> None:
        summary = summarize([])
        self.assertEqual(summary["records"], 0)
        self.assertEqual(summary["errors"], 0)


if __name__ == "__main__":
    unittest.main()
