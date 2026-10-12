"""Writer · 方案撰写（单线程，全系统唯一允许「写入」的环节）。

为什么只有它能写：并行的 agent 各自携带隐含决策（选哪个模型、用哪种部署方式），
如果让它们各自产出成品片段，最后拼起来会互相打架。工程结论是让**写入保持单线程**，
其他 agent 只贡献信息、不产生动作。

另一条纪律：每条技术断言都必须携带 `evidence_ref`。这不是提示词里的请求，
而是契约层强制的 —— 没有引用的断言在 SolutionDraft 里就不合法。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from arksol.agents.base import Agent
from arksol.state import RunState

_COMMON_RULES = (
    "输出字段：\n"
    "- summary：一句话方案概述（说明选了哪个档位的模型、怎么部署）\n"
    "- model_choice：推荐的模型档位与理由\n"
    "- deployment：部署方式\n"
    "- claims：技术断言数组。**每条断言必须携带 evidence_ref**，"
    "它只能取两类值：证据表里真实存在的证据 id，或 EV-NONE\n"
    "- followup_questions：这份方案还需要向客户追问什么；没有就输出空数组\n\n"
    "红线：**不要编造 evidence_ref**。指向一个不存在的证据，比老老实实标 EV-NONE 严重得多 —— "
    "前者伪装成有依据。"
)


class MemoryPlanAgent(Agent):
    """基线 A：不看知识库，直接凭自己的知识出方案。

    它仍然被要求逐条标注断言来源 —— 只是它没有来源可标，于是全部落到 EV-NONE。
    这样审计器就能用**同一把尺子**量它，而不是靠人工判断「它看起来编了」。
    """

    name = "plan_from_memory"
    tag = "single"
    schema_name = "SolutionDraft"
    max_tokens = 1600

    def __init__(self, provider, *, requirement: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(provider, **kwargs)
        self._requirement = requirement

    def system_prompt(self) -> str:
        return (
            "你是大模型平台的方案专家。本轮**没有检索工具**，你只能凭自己的知识作答。"
            "你只输出 JSON。"
        )

    def instruction(self) -> str:
        return (
            "请针对下面的结构化需求，直接给出技术方案。\n\n"
            + _COMMON_RULES
            + "\n由于你没有检索结果，**所有断言都必须标 EV-NONE**，不要编造任何证据编号。"
        )

    def payload(self, state: RunState) -> Dict[str, Any]:
        return {"requirement": self._requirement, "evidence_items": []}


class WriterAgent(Agent):
    """B / C 共用的写作环节：有证据，且必须挂到证据上。"""

    name = "writer"
    tag = "writer"
    schema_name = "SolutionDraft"
    max_tokens = 1600

    def __init__(
        self,
        provider,
        *,
        requirement: Dict[str, Any],
        evidence_items: Sequence[Dict[str, Any]] = (),
        revision_notes: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(provider, **kwargs)
        self._requirement = requirement
        self._evidence_items = list(evidence_items)
        self._revision_notes = list(revision_notes or [])

    def system_prompt(self) -> str:
        return (
            "你是大模型平台的产品解决方案工程师。你的每一条技术断言都必须来自给定证据，"
            "**不允许使用证据之外的知识**。你只输出 JSON。"
        )

    def instruction(self) -> str:
        return (
            "请基于下方 JSON 里的 requirement（客户需求）与 evidence_items（可引用的全部证据）"
            "写一份技术方案草稿。\n\n"
            + _COMMON_RULES
            + "\n另外：方案必须逐条回应客户提出的每一条约束（尤其是 critical 的）。"
        )

    def payload(self, state: RunState) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "requirement": self._requirement,
            "evidence_items": self._evidence_items,
        }
        if self._revision_notes:
            payload["revision_notes"] = self._revision_notes
        return payload
