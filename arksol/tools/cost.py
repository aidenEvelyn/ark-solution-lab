"""成本换算：token → 元。

⚠️ 三件事必须分清，否则数字会骗人：

1. **精确 vs 估算**：真实调用返回的 `usage.prompt_tokens / completion_tokens` 是精确的；
   没有 usage 时按字符数折算，只能当**数量级**。两者在报告里必须分开标注。
2. **单价的可信度**：单价来自知识库，而知识库里的价格主要来自公开文档的**转述**，
   可信度标为「中」。所以最终金额是「按公开单价的估算」，**不是账单**。
3. **未计入的部分**：缓存命中折扣、批量推理折扣、免费额度抵扣都没有算进去。
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def is_cjk(char: str) -> bool:
    code = ord(char)
    return 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。

    经验口径：CJK 字符约 1 token/字，其余字符约 4 字符/token。
    ⚠️ 这是估算 —— 真实 token 数以提供方返回的 usage 为准。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if is_cjk(ch))
    other = len(text) - cjk
    return max(1, int(round(cjk + other / 4.0)))


def cost_cny(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    price_in_cny_per_mtok: float,
    price_out_cny_per_mtok: float,
) -> float:
    """按「元 / 百万 tokens」单价换算金额。"""
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("token 数不能为负")
    return (prompt_tokens / 1_000_000.0) * price_in_cny_per_mtok + (
        completion_tokens / 1_000_000.0
    ) * price_out_cny_per_mtok


def cost_from_usage(usage: Dict[str, int], price: Dict[str, float]) -> float:
    """从提供方返回的 usage 直接换算。"""
    return cost_cny(
        int(usage.get("prompt_tokens", 0)),
        int(usage.get("completion_tokens", 0)),
        price_in_cny_per_mtok=float(price["price_in_cny_per_mtok"]),
        price_out_cny_per_mtok=float(price["price_out_cny_per_mtok"]),
    )


def price_from_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从知识库条目的 facts 里取单价。

    取不到就返回 None（**不猜**）—— 上层据此把成本标成「未计入」，
    而不是编一个数出来。这是这个项目里「不许编」的又一处落地。
    """
    facts = entry.get("facts") or {}
    price_in = facts.get("price_in_cny_per_mtok")
    price_out = facts.get("price_out_cny_per_mtok")
    if price_in is None or price_out is None:
        return None
    return {
        "entry_id": entry.get("id"),
        "price_in_cny_per_mtok": float(price_in),
        "price_out_cny_per_mtok": float(price_out),
        "confidence": facts.get("price_confidence", "未标注"),
        "note": facts.get("price_note", ""),
    }


def format_cny(value: float) -> str:
    """金额展示：小到 0.0001 元也要看得出来，不要四舍五入成 0。"""
    if value == 0:
        return "0 元"
    if abs(value) < 0.01:
        return f"{value:.6f} 元"
    if abs(value) < 1:
        return f"{value:.4f} 元"
    return f"{value:.2f} 元"
