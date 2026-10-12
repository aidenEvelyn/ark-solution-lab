"""成本换算测试。

这些断言保护的是一个容易被悄悄改坏的约定：
**算不出单价时返回 None，而不是猜一个数出来。**
"""

from __future__ import annotations

import unittest

from arksol.kb import load_default_kb
from arksol.tools.cost import cost_cny, cost_from_usage, estimate_tokens, format_cny, is_cjk, price_from_entry


class TestEstimateTokens(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)

    def test_pure_chinese(self) -> None:
        self.assertEqual(estimate_tokens("你好世界"), 4)

    def test_pure_ascii(self) -> None:
        self.assertEqual(estimate_tokens("abcd"), 1)

    def test_mixed(self) -> None:
        self.assertEqual(estimate_tokens("你好abcd"), 3)

    def test_nonempty_never_zero(self) -> None:
        self.assertGreaterEqual(estimate_tokens("a"), 1)

    def test_is_cjk(self) -> None:
        self.assertTrue(is_cjk("上"))
        self.assertFalse(is_cjk("a"))


class TestCostMath(unittest.TestCase):
    def test_one_million_each(self) -> None:
        self.assertAlmostEqual(
            cost_cny(1_000_000, 1_000_000, price_in_cny_per_mtok=3.0, price_out_cny_per_mtok=15.0), 18.0
        )

    def test_input_only(self) -> None:
        self.assertAlmostEqual(cost_cny(500_000, 0, price_in_cny_per_mtok=3.0, price_out_cny_per_mtok=15.0), 1.5)

    def test_zero(self) -> None:
        self.assertEqual(cost_cny(0, 0, price_in_cny_per_mtok=3.0, price_out_cny_per_mtok=15.0), 0.0)

    def test_small_amount_not_rounded_to_zero(self) -> None:
        value = cost_cny(1_000, 500, price_in_cny_per_mtok=3.0, price_out_cny_per_mtok=15.0)
        self.assertGreater(value, 0.0)

    def test_negative_tokens_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cost_cny(-1, 0, price_in_cny_per_mtok=3.0, price_out_cny_per_mtok=15.0)

    def test_from_usage(self) -> None:
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
        price = {"price_in_cny_per_mtok": 3.0, "price_out_cny_per_mtok": 15.0}
        self.assertAlmostEqual(cost_from_usage(usage, price), 3.0)


class TestPriceFromEntry(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = load_default_kb()

    def test_reads_price_with_confidence(self) -> None:
        price = price_from_entry(self.kb.get("model.doubao-seed-2-1-turbo"))
        self.assertIsNotNone(price)
        self.assertEqual(price["price_in_cny_per_mtok"], 3.0)
        self.assertEqual(price["confidence"], "低")

    def test_returns_none_when_price_unknown(self) -> None:
        # 这一条正是「不许编」的证据：知识库里没查到单价，就不能凭空给一个
        self.assertIsNone(price_from_entry(self.kb.get("model.doubao-seed-evolving")))

    def test_returns_none_without_facts(self) -> None:
        self.assertIsNone(price_from_entry({"id": "x"}))
        self.assertIsNone(price_from_entry({"id": "x", "facts": {}}))


class TestFormatCny(unittest.TestCase):
    def test_zero(self) -> None:
        self.assertEqual(format_cny(0.0), "0 元")

    def test_very_small(self) -> None:
        self.assertEqual(format_cny(0.0001), "0.000100 元")

    def test_sub_yuan(self) -> None:
        self.assertEqual(format_cny(0.5), "0.5000 元")

    def test_normal(self) -> None:
        self.assertEqual(format_cny(12.4), "12.40 元")


if __name__ == "__main__":
    unittest.main()
