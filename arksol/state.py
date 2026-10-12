"""外置状态与交接单。

为什么状态不放在上下文里：

1. **成本**：无状态 API 每步重发全部历史，累计输入随步数近似平方增长。
2. **正确性**：任务态与执行历史共用一份不断增长的上下文时，早期的错误判断会被带进
   后续决策并累积，而不是自愈。把状态搬出去，可以让每个环节只拿到**它需要的那一份**。

交接单（handoff note）是这套东西的载体：下一个环节读单据，而不是读全部历史。
其中最有价值的一栏是 `pitfalls` —— **已经试过、并且知道为什么走不通的路**。
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class Attempt:
    """一次尝试的记录。既是审计线索，也是重复动作熔断的依据。"""

    stage: str
    round: int
    outcome: str  # ok | contract_error | revise | error | budget_stop
    reason: str = ""

    @property
    def fingerprint(self) -> str:
        return f"{self.stage}:{self.outcome}"


@dataclass
class RunState:
    user_input: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=_now_iso)

    stage: str = "start"
    requirement: Optional[Dict[str, Any]] = None
    followup: Optional[Dict[str, Any]] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    draft: Optional[Dict[str, Any]] = None
    critiques: List[Dict[str, Any]] = field(default_factory=list)
    attempts: List[Attempt] = field(default_factory=list)
    stop_reason: Optional[str] = None

    # -- 写入 ---------------------------------------------------------------------------

    def record_attempt(self, *, stage: str, round: int, outcome: str, reason: str = "") -> Attempt:
        attempt = Attempt(stage=stage, round=round, outcome=outcome, reason=reason)
        self.attempts.append(attempt)
        return attempt

    def set_stage(self, stage: str) -> None:
        self.stage = stage

    # -- 读取 ---------------------------------------------------------------------------

    @property
    def finished(self) -> bool:
        if self.stop_reason:
            return True
        if self.followup is not None:
            return True  # 信息不足 → 出追问清单即结束，不硬编方案
        if self.draft is not None and any(c.get("verdict") == "pass" for c in self.critiques):
            return True
        return False

    def evidence_items(self) -> List[Dict[str, Any]]:
        """把多个取证 agent 的证据包摊平成一个列表。"""
        items: List[Dict[str, Any]] = []
        for bundle in self.evidence:
            items.extend(bundle.get("items", []))
        return items

    def evidence_by_id(self) -> Dict[str, Dict[str, Any]]:
        return {item.get("id", ""): item for item in self.evidence_items()}

    def source_ids(self) -> List[str]:
        return sorted({item.get("source_id", "") for item in self.evidence_items() if item.get("source_id")})

    def failed_fingerprints(self) -> Dict[str, int]:
        """每个「环节:结果」出现过几次 —— 用来发现「同一个失败动作在反复重试」。"""
        counts: Dict[str, int] = {}
        for attempt in self.attempts:
            if attempt.outcome != "ok":
                counts[attempt.fingerprint] = counts.get(attempt.fingerprint, 0) + 1
        return counts

    def last_critique(self) -> Optional[Dict[str, Any]]:
        return self.critiques[-1] if self.critiques else None

    # -- 交接单 -------------------------------------------------------------------------

    def handoff_note(
        self,
        *,
        stage: str,
        task: str,
        needs: List[str],
        todo: List[str],
        acceptance: List[str],
        decisions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """给下一个环节的单据。

        `pitfalls` 不是装饰：它是「已试过且失败 + 为什么」的指纹集合，
        下一个环节据此不会再去撞同一堵墙。这正是长链条里最容易丢的信息。
        """
        return {
            "stage": stage,
            "task": task,
            "done": {
                "requirement": self.requirement is not None,
                "followup": self.followup is not None,
                "evidence_bundles": len(self.evidence),
                "evidence_items": len(self.evidence_items()),
                "draft": self.draft is not None,
                "critiques": len(self.critiques),
            },
            "decisions": decisions or [],
            "pitfalls": [
                {"stage": a.stage, "round": a.round, "outcome": a.outcome, "reason": a.reason}
                for a in self.attempts
                if a.outcome != "ok"
            ],
            "needs": needs,
            "todo": todo,
            "acceptance": acceptance,
        }

    # -- 序列化 -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["finished"] = self.finished
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunState":
        payload = dict(data)
        payload.pop("finished", None)
        attempts = [Attempt(**a) for a in payload.pop("attempts", [])]
        state = cls(**payload)
        state.attempts = attempts
        return state
