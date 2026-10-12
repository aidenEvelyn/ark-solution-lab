"""预算、计量与熔断。

三类失败必须有硬上限，靠模型自觉是没用的：

1. **成本非线性**：无状态 API 每步重发全部历史，累计输入随步数近似平方增长。
2. **沉默失败**：日志里没有 error，轨迹看起来还在活动，实际在重复同一个无效动作。
3. **并行记账竞争**：取证环节是真的多线程，计数器和文件写入必须加锁。

所以这里给的是三重上限（token / 费用 / 步数）+ 重复动作指纹 + 80% 提前告警，
外加一个累加式的 Meter —— 把「这一次运行花了多少」和「还能花多少」分开记账。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from arksol.errors import BudgetExceeded, RepeatActionDetected


@dataclass
class Budget:
    """一次运行的硬上限。默认值按「一次演示跑得起」设定，不是为了生产。"""

    max_tokens: int = 200_000
    max_cost_cny: float = 5.0
    max_steps: int = 200
    warn_ratio: float = 0.8
    max_repeat: int = 3


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Meter:
    """一次运行的花费流水。与 BudgetGuard 的分工：

    - Meter 记账（我花了多少）
    - BudgetGuard 管限额（我还剩多少、什么时候必须停）
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_cny: float = 0.0
    estimated_calls: int = 0
    by_agent: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, agent: str, response: Any) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += int(getattr(response, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(response, "completion_tokens", 0) or 0)
            self.cost_cny += float(getattr(response, "cost_cny", 0.0) or 0.0)
            if getattr(response, "usage_is_estimated", True):
                self.estimated_calls += 1
            self.by_agent[agent] = self.by_agent.get(agent, 0) + 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def note_error(self, message: str) -> None:
        with self._lock:
            self.errors.append(message)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_cny": round(self.cost_cny, 6),
            "estimated_calls": self.estimated_calls,
            "by_agent": dict(self.by_agent),
            "errors": list(self.errors),
        }


@dataclass
class BudgetGuard:
    """预算守卫。每次调用模型前后各记一次账。并行场景下方法是线程安全的。"""

    budget: Budget = field(default_factory=Budget)
    tokens_used: int = 0
    cost_used_cny: float = 0.0
    steps_used: int = 0
    warnings: List[str] = field(default_factory=list)
    _fingerprints: Dict[str, int] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # -- 记账 ---------------------------------------------------------------------------

    def charge(self, *, tokens: int = 0, cost_cny: float = 0.0) -> None:
        """记一次账；超限立即抛 BudgetExceeded（不返回状态码，避免调用方忘记检查）。"""
        with self._lock:
            self.tokens_used += max(0, int(tokens))
            self.cost_used_cny += max(0.0, float(cost_cny))
            self._check_limits()

    def step(self) -> None:
        """记一步。"""
        with self._lock:
            self.steps_used += 1
            if self.steps_used > self.budget.max_steps:
                raise BudgetExceeded("steps", self.steps_used, self.budget.max_steps)

    def note_action(self, fingerprint: str) -> None:
        """记一个动作指纹（例如 'retrieve:模型选型'）。

        同一个指纹出现次数达到上限即熔断 —— 这是「相同工具+相同参数反复重试」的守门人。
        """
        with self._lock:
            count = self._fingerprints.get(fingerprint, 0) + 1
            self._fingerprints[fingerprint] = count
            if count >= self.budget.max_repeat:
                raise RepeatActionDetected(fingerprint, count)

    # -- 状态 ---------------------------------------------------------------------------

    def _check_limits(self) -> None:
        if self.tokens_used > self.budget.max_tokens:
            raise BudgetExceeded("tokens", self.tokens_used, self.budget.max_tokens)
        if self.cost_used_cny > self.budget.max_cost_cny:
            raise BudgetExceeded("cost_cny", round(self.cost_used_cny, 4), self.budget.max_cost_cny)

    def _usage_ratio(self, used: float, limit: float) -> float:
        return 0.0 if limit <= 0 else used / limit

    def check_warnings(self) -> List[str]:
        """返回并累积「用掉 80% 仍未收敛」类告警。不抛异常 —— 告警不是熔断。"""
        with self._lock:
            ratios = {
                "tokens": self._usage_ratio(self.tokens_used, self.budget.max_tokens),
                "cost_cny": self._usage_ratio(self.cost_used_cny, self.budget.max_cost_cny),
                "steps": self._usage_ratio(self.steps_used, self.budget.max_steps),
            }
            for kind, ratio in ratios.items():
                if ratio >= self.budget.warn_ratio:
                    message = f"{kind} 已用 {ratio:.0%}（{self.budget.warn_ratio:.0%} 告警线）"
                    if message not in self.warnings:
                        self.warnings.append(message)
            return list(self.warnings)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.budget.max_tokens - self.tokens_used)

    @property
    def repeat_counts(self) -> Dict[str, int]:
        return dict(self._fingerprints)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "tokens_used": self.tokens_used,
                "cost_used_cny": round(self.cost_used_cny, 6),
                "steps_used": self.steps_used,
                "limits": {
                    "max_tokens": self.budget.max_tokens,
                    "max_cost_cny": self.budget.max_cost_cny,
                    "max_steps": self.budget.max_steps,
                },
                "warnings": list(self.warnings),
                "repeat_counts": dict(self._fingerprints),
            }


def safe_charge(guard: Optional[BudgetGuard], *, tokens: int = 0, cost_cny: float = 0.0) -> Optional[str]:
    """给「没有预算守卫也要能跑」的场景用的记账函数。

    返回熔断原因；没有守卫或未超限则返回 None。调用方选择停下还是继续 ——
    但**必须显式选择**，不能静默忽略。
    """
    if guard is None:
        return None
    try:
        guard.charge(tokens=tokens, cost_cny=cost_cny)
    except BudgetExceeded as exc:
        return str(exc)
    return None
