"""独立审计器测试。

这一层的输出要同时满足两个苛刻要求：
1. 是确定性规则（能复现、能复核）；
2. 本身就是一份合法的 Critique 契约（否则评测台的裁判自己就不合规）。
"""

from __future__ import annotations

import unittest

from arksol.contracts import validate_named
from arksol.verify import (
    DEFAULT_CONSTRAINT_KEYWORDS,
    NO_EVIDENCE_MARK,
    check_constraints,
    classify_claims,
    draft_text,
    keywords_for,
    review,
)


def make_draft(claims, *, summary="默认摘要", deployment="公有云在线推理", followups=None) -> dict:
    return {
        "summary": summary,
        "model_choice": "doubao-seed-2-1-turbo",
        "deployment": deployment,
        "claims": claims,
        "followup_questions": list(followups or []),
    }


def make_claim(text="这是一条断言", ref="EV-001", kind="capability") -> dict:
    return {"text": text, "evidence_ref": ref, "kind": kind}


def requirement_with(kinds) -> dict:
    detail = {
        "compliance": "数据不能出内网",
        "latency": "首字延迟 2 秒内",
        "cost": "每月预算两千",
        "concurrency": "峰值 QPS 50",
        "other": "客户的某个特殊要求",
    }
    return {
        "business_scenario": "内部知识库问答",
        "capabilities": ["知识库问答"],
        "constraints": [{"kind": k, "detail": detail[k], "critical": True} for k in kinds],
        "open_questions": [],
        "sufficient": True,
    }


class TestDraftText(unittest.TestCase):
    def test_includes_all_human_visible_parts(self) -> None:
        text = draft_text(make_draft([make_claim(text="断言正文")], summary="摘要正文", followups=["还需要问什么"]))
        for fragment in ("摘要正文", "断言正文", "还需要问什么", "doubao-seed-2-1-turbo"):
            self.assertIn(fragment, text)

    def test_tolerates_missing_fields(self) -> None:
        self.assertEqual(draft_text({}), "")


class TestClassifyClaims(unittest.TestCase):
    def test_all_supported(self) -> None:
        result = classify_claims(make_draft([make_claim(ref="EV-001"), make_claim(ref="EV-002")]), {"EV-001", "EV-002"})
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["unsupported_count"], 0)
        self.assertEqual(result["unsupported_rate"], 0.0)

    def test_mixed_buckets_are_separated(self) -> None:
        draft = make_draft(
            [
                make_claim(ref="EV-001"),
                make_claim(ref=NO_EVIDENCE_MARK),
                make_claim(ref="EV-999"),
            ]
        )
        result = classify_claims(draft, {"EV-001"})
        self.assertEqual(len(result["supported"]), 1)
        self.assertEqual(len(result["no_evidence"]), 1)
        self.assertEqual(len(result["dangling"]), 1)
        self.assertEqual(result["unsupported_count"], 2)
        self.assertAlmostEqual(result["unsupported_rate"], 2 / 3)

    def test_dangling_is_distinguished_from_no_evidence(self) -> None:
        # 「引用了不存在的来源」比「老实说没证据」更严重，因为它伪装成了有依据
        result = classify_claims(make_draft([make_claim(ref="EV-999")]), {"EV-001"})
        self.assertEqual(len(result["dangling"]), 1)
        self.assertEqual(len(result["no_evidence"]), 0)

    def test_empty_draft(self) -> None:
        result = classify_claims(make_draft([]), set())
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["unsupported_rate"], 0.0)


class TestConstraintCoverage(unittest.TestCase):
    def test_all_covered(self) -> None:
        draft = make_draft([make_claim()], summary="方案会满足数据合规与延迟要求", deployment="私有化部署")
        result = check_constraints(requirement_with(["compliance", "latency"]), draft)
        self.assertEqual(result["judgeable"], 2)
        self.assertEqual(result["coverage_rate"], 1.0)
        self.assertTrue(all(c["matched_keyword"] for c in result["covered"]))

    def test_missed_constraint(self) -> None:
        draft = make_draft([make_claim()], summary="方案整体可行", deployment="公有云在线推理")
        result = check_constraints(requirement_with(["compliance"]), draft)
        self.assertEqual(len(result["missed"]), 1)
        self.assertEqual(result["coverage_rate"], 0.0)
        self.assertEqual(result["missed"][0]["kind"], "compliance")

    def test_unjudgeable_kind_excluded_from_denominator(self) -> None:
        draft = make_draft([make_claim()], summary="方案满足延迟要求")
        result = check_constraints(requirement_with(["latency", "other"]), draft)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["judgeable"], 1)
        self.assertEqual(len(result["unjudgeable"]), 1)
        self.assertEqual(result["coverage_rate"], 1.0)

    def test_no_constraints_counts_as_full_coverage(self) -> None:
        result = check_constraints({"constraints": []}, make_draft([]))
        self.assertEqual(result["judgeable"], 0)
        self.assertEqual(result["coverage_rate"], 1.0)

    def test_keyword_map_override(self) -> None:
        requirement = requirement_with(["compliance"])
        draft = make_draft([make_claim()], summary="本方案采用银河级加密")
        custom = {"compliance": ["银河级加密"]}
        self.assertEqual(len(check_constraints(requirement, draft, keyword_map=custom)["covered"]), 1)

    def test_keywords_for_defaults(self) -> None:
        self.assertEqual(keywords_for("compliance"), DEFAULT_CONSTRAINT_KEYWORDS["compliance"])
        self.assertEqual(keywords_for("完全不存在的种类"), [])


class TestReview(unittest.TestCase):
    def test_clean_draft_passes(self) -> None:
        draft = make_draft([make_claim(ref="EV-001")], summary="满足数据合规要求", deployment="私有化")
        result = review(draft, evidence_ids={"EV-001"}, requirement=requirement_with(["compliance"]))
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["checked_claims"], 1)

    def test_dangling_claim_triggers_revise(self) -> None:
        draft = make_draft([make_claim(ref="EV-999")])
        result = review(draft, evidence_ids={"EV-001"})
        self.assertEqual(result["verdict"], "revise")
        self.assertEqual(result["issues"][0]["type"], "unsupported_claim")
        self.assertIn("EV-999", result["issues"][0]["detail"])

    def test_no_evidence_claim_triggers_revise(self) -> None:
        draft = make_draft([make_claim(ref=NO_EVIDENCE_MARK)])
        result = review(draft, evidence_ids={"EV-001"})
        self.assertEqual(result["verdict"], "revise")
        self.assertIn("没有证据支撑", result["issues"][0]["detail"])

    def test_missed_constraint_triggers_revise(self) -> None:
        draft = make_draft([make_claim(ref="EV-001")], summary="方案整体可行")
        result = review(draft, evidence_ids={"EV-001"}, requirement=requirement_with(["compliance"]))
        types = [issue["type"] for issue in result["issues"]]
        self.assertIn("uncovered_constraint", types)

    def test_review_output_is_a_valid_critique(self) -> None:
        draft = make_draft([make_claim(ref="EV-999")], summary="方案整体可行")
        result = review(draft, evidence_ids=set(), requirement=requirement_with(["compliance"]))
        validate_named(result, "Critique")  # 裁判自己必须合规

    def test_clean_output_is_also_valid_critique(self) -> None:
        validate_named(review(make_draft([make_claim(ref="EV-1")]), evidence_ids={"EV-1"}), "Critique")

    def test_extra_issues_are_merged(self) -> None:
        result = review(
            make_draft([make_claim(ref="EV-1")]),
            evidence_ids={"EV-1"},
            extra_issues=[{"type": "other", "detail": "承诺了没有依据的上线时间"}],
        )
        self.assertEqual(result["verdict"], "revise")
        self.assertEqual(result["issues"][0]["type"], "other")

    def test_works_without_requirement(self) -> None:
        result = review(make_draft([make_claim(ref="EV-1")]), evidence_ids={"EV-1"})
        self.assertEqual(result["verdict"], "pass")


if __name__ == "__main__":
    unittest.main()
