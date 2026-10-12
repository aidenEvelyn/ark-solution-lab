"""外置状态与交接单测试。"""

from __future__ import annotations

import unittest

from arksol.state import RunState


def bundle(topic: str, item_ids: list[str]) -> dict:
    return {
        "topic": topic,
        "items": [
            {
                "id": item_id,
                "topic": topic,
                "text": f"{topic} 的证据正文 {item_id}",
                "source_id": "S01",
                "confidence": "中",
            }
            for item_id in item_ids
        ],
        "missing": [],
    }


class TestLifecycle(unittest.TestCase):
    def test_new_state(self) -> None:
        state = RunState(user_input="想做个内部知识库问答")
        self.assertEqual(state.stage, "start")
        self.assertFalse(state.finished)
        self.assertTrue(state.run_id)

    def test_followup_finishes_run(self) -> None:
        state = RunState(user_input="想做个问答")
        state.followup = {"questions": [{"question": "数据能出内网吗", "why_critical": "决定部署方式"}], "assumptions": []}
        self.assertTrue(state.finished)

    def test_draft_alone_is_not_finished(self) -> None:
        state = RunState(user_input="x")
        state.draft = {"summary": "草稿"}
        self.assertFalse(state.finished)

    def test_draft_with_passing_critique_finishes(self) -> None:
        state = RunState(user_input="x")
        state.draft = {"summary": "草稿"}
        state.critiques.append({"verdict": "pass", "issues": [], "checked_claims": 3})
        self.assertTrue(state.finished)

    def test_draft_with_revise_is_not_finished(self) -> None:
        state = RunState(user_input="x")
        state.draft = {"summary": "草稿"}
        state.critiques.append({"verdict": "revise", "issues": [], "checked_claims": 3})
        self.assertFalse(state.finished)

    def test_stop_reason_finishes_run(self) -> None:
        state = RunState(user_input="x")
        state.stop_reason = "预算熔断：tokens"
        self.assertTrue(state.finished)


class TestAttempts(unittest.TestCase):
    def test_record_attempt(self) -> None:
        state = RunState(user_input="x")
        attempt = state.record_attempt(stage="writer", round=1, outcome="revise", reason="有 2 条无依据断言")
        self.assertEqual(attempt.fingerprint, "writer:revise")
        self.assertEqual(len(state.attempts), 1)

    def test_failed_fingerprints_counts_only_failures(self) -> None:
        state = RunState(user_input="x")
        state.record_attempt(stage="intake", round=1, outcome="ok")
        state.record_attempt(stage="writer", round=1, outcome="revise")
        state.record_attempt(stage="writer", round=2, outcome="revise")
        self.assertEqual(state.failed_fingerprints(), {"writer:revise": 2})


class TestEvidenceAggregation(unittest.TestCase):
    def test_items_flattened_in_order(self) -> None:
        state = RunState(user_input="x")
        state.evidence = [bundle("模型选型", ["EV-1", "EV-2"]), bundle("合规", ["EV-3"])]
        self.assertEqual([i["id"] for i in state.evidence_items()], ["EV-1", "EV-2", "EV-3"])

    def test_index_by_id(self) -> None:
        state = RunState(user_input="x")
        state.evidence = [bundle("模型选型", ["EV-1"])]
        self.assertIn("EV-1", state.evidence_by_id())

    def test_source_ids_unique_and_sorted(self) -> None:
        state = RunState(user_input="x")
        state.evidence = [bundle("a", ["EV-1"]), bundle("b", ["EV-2"])]
        self.assertEqual(state.source_ids(), ["S01"])

    def test_no_evidence(self) -> None:
        state = RunState(user_input="x")
        self.assertEqual(state.evidence_items(), [])
        self.assertEqual(state.source_ids(), [])

    def test_last_critique(self) -> None:
        state = RunState(user_input="x")
        self.assertIsNone(state.last_critique())
        state.critiques = [{"verdict": "revise"}, {"verdict": "pass"}]
        self.assertEqual(state.last_critique(), {"verdict": "pass"})


class TestHandoffNote(unittest.TestCase):
    def test_note_shape(self) -> None:
        state = RunState(user_input="x")
        note = state.handoff_note(
            stage="writer",
            task="写出方案草稿",
            needs=["requirement", "evidence"],
            todo=["每条断言挂 evidence_ref"],
            acceptance=["契约校验通过", "所有 critical 约束被回应"],
        )
        for key in ("stage", "task", "done", "decisions", "pitfalls", "needs", "todo", "acceptance"):
            self.assertIn(key, note)

    def test_done_counts_reflect_state(self) -> None:
        state = RunState(user_input="x")
        state.requirement = {"business_scenario": "x"}
        state.evidence = [bundle("a", ["EV-1", "EV-2"])]
        note = state.handoff_note(stage="writer", task="t", needs=[], todo=[], acceptance=[])
        self.assertTrue(note["done"]["requirement"])
        self.assertEqual(note["done"]["evidence_items"], 2)
        self.assertFalse(note["done"]["draft"])

    def test_pitfalls_carry_failed_attempts(self) -> None:
        state = RunState(user_input="x")
        state.record_attempt(stage="intake", round=1, outcome="contract_error", reason="缺 open_questions")
        state.record_attempt(stage="intake", round=2, outcome="ok")
        note = state.handoff_note(stage="evidence", task="t", needs=[], todo=[], acceptance=[])
        self.assertEqual(len(note["pitfalls"]), 1)
        self.assertEqual(note["pitfalls"][0]["outcome"], "contract_error")


class TestSerialization(unittest.TestCase):
    def test_round_trip(self) -> None:
        state = RunState(user_input="客户原话", run_id="fixed123")
        state.requirement = {"business_scenario": "知识库问答"}
        state.evidence = [bundle("模型选型", ["EV-1"])]
        state.draft = {"summary": "建议直接用长上下文模型吃下全部文档。"}
        state.critiques.append({"verdict": "pass", "issues": [], "checked_claims": 1})
        state.record_attempt(stage="writer", round=1, outcome="ok")

        restored = RunState.from_dict(state.to_dict())

        self.assertEqual(restored.run_id, "fixed123")
        self.assertEqual(restored.evidence_items()[0]["id"], "EV-1")
        self.assertEqual(len(restored.attempts), 1)
        self.assertTrue(restored.finished)

    def test_to_dict_has_finished_flag(self) -> None:
        state = RunState(user_input="x")
        self.assertIn("finished", state.to_dict())


if __name__ == "__main__":
    unittest.main()
