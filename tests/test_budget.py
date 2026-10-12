"""预算与熔断测试。

两类失败在这里被挡住：**成本失控** 与 **同一个动作反复重试的沉默失败**。
"""

from __future__ import annotations

import unittest

from arksol.budget import Budget, BudgetGuard, safe_charge
from arksol.errors import BudgetExceeded, RepeatActionDetected


class TestCharge(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = BudgetGuard(Budget(max_tokens=1000, max_cost_cny=1.0, max_steps=5))

    def test_within_limits(self) -> None:
        self.guard.charge(tokens=500, cost_cny=0.5)
        self.assertEqual(self.guard.tokens_used, 500)
        self.assertAlmostEqual(self.guard.cost_used_cny, 0.5)

    def test_tokens_over_limit(self) -> None:
        with self.assertRaises(BudgetExceeded) as ctx:
            self.guard.charge(tokens=1001)
        self.assertEqual(ctx.exception.kind, "tokens")

    def test_cost_over_limit(self) -> None:
        with self.assertRaises(BudgetExceeded) as ctx:
            self.guard.charge(cost_cny=1.01)
        self.assertEqual(ctx.exception.kind, "cost_cny")

    def test_accumulated_charge_trips_limit(self) -> None:
        self.guard.charge(tokens=600)
        with self.assertRaises(BudgetExceeded):
            self.guard.charge(tokens=600)

    def test_negative_charge_ignored(self) -> None:
        self.guard.charge(tokens=-100, cost_cny=-1.0)
        self.assertEqual(self.guard.tokens_used, 0)
        self.assertEqual(self.guard.cost_used_cny, 0.0)

    def test_steps_limit(self) -> None:
        for _ in range(5):
            self.guard.step()
        with self.assertRaises(BudgetExceeded) as ctx:
            self.guard.step()
        self.assertEqual(ctx.exception.kind, "steps")

    def test_remaining_tokens(self) -> None:
        self.guard.charge(tokens=400)
        self.assertEqual(self.guard.remaining_tokens, 600)


class TestWarnings(unittest.TestCase):
    def test_warning_at_threshold(self) -> None:
        guard = BudgetGuard(Budget(max_tokens=1000, warn_ratio=0.8))
        guard.charge(tokens=800)
        warnings = guard.check_warnings()
        self.assertTrue(any("tokens" in w for w in warnings))

    def test_no_warning_below_threshold(self) -> None:
        guard = BudgetGuard(Budget(max_tokens=1000, warn_ratio=0.8))
        guard.charge(tokens=700)
        self.assertEqual(guard.check_warnings(), [])

    def test_warning_not_duplicated(self) -> None:
        guard = BudgetGuard(Budget(max_tokens=1000, warn_ratio=0.8))
        guard.charge(tokens=800)
        guard.check_warnings()
        guard.check_warnings()
        self.assertEqual(len([w for w in guard.warnings if "tokens" in w]), 1)

    def test_warning_does_not_raise(self) -> None:
        guard = BudgetGuard(Budget(max_tokens=1000, max_cost_cny=1.0, warn_ratio=0.8))
        guard.charge(tokens=900, cost_cny=0.9)
        guard.check_warnings()  # 告警不是熔断


class TestRepeatAction(unittest.TestCase):
    def test_third_identical_action_trips(self) -> None:
        guard = BudgetGuard(Budget(max_repeat=3))
        guard.note_action("retrieve:模型选型")
        guard.note_action("retrieve:模型选型")
        with self.assertRaises(RepeatActionDetected) as ctx:
            guard.note_action("retrieve:模型选型")
        self.assertEqual(ctx.exception.count, 3)

    def test_different_actions_are_independent(self) -> None:
        guard = BudgetGuard(Budget(max_repeat=3))
        guard.note_action("a")
        guard.note_action("b")
        guard.note_action("c")
        self.assertEqual(guard.repeat_counts, {"a": 1, "b": 1, "c": 1})

    def test_custom_limit(self) -> None:
        guard = BudgetGuard(Budget(max_repeat=2))
        guard.note_action("x")
        with self.assertRaises(RepeatActionDetected):
            guard.note_action("x")


class TestSnapshot(unittest.TestCase):
    def test_snapshot_shape(self) -> None:
        guard = BudgetGuard(Budget(max_tokens=100))
        guard.charge(tokens=10)
        guard.step()
        snapshot = guard.snapshot()
        self.assertEqual(snapshot["tokens_used"], 10)
        self.assertEqual(snapshot["steps_used"], 1)
        self.assertIn("limits", snapshot)
        self.assertIn("warnings", snapshot)


class TestSafeCharge(unittest.TestCase):
    def test_without_guard(self) -> None:
        self.assertIsNone(safe_charge(None, tokens=10))

    def test_ok(self) -> None:
        guard = BudgetGuard(Budget(max_tokens=100))
        self.assertIsNone(safe_charge(guard, tokens=10))

    def test_returns_reason_instead_of_raising(self) -> None:
        guard = BudgetGuard(Budget(max_tokens=100))
        reason = safe_charge(guard, tokens=500)
        self.assertIsNotNone(reason)
        self.assertIn("tokens", reason or "")


if __name__ == "__main__":
    unittest.main()
