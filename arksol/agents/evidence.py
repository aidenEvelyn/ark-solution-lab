"""Evidence · 取证（可并行、多实例、严格只读）。

这是本项目**唯一使用并行多智能体**的地方，三条独立证据指向同一个结论：

1. 多智能体只在「可高度并行、信息超出单窗口、需要对接大量工具」的任务上有明确收益；
2. 这类子任务最安全的形态是**只读** —— 它只贡献信息、不产生动作，
   因此不存在「两个 agent 各自做了互相矛盾的隐含决定」的问题；
3. 多智能体的 token 消耗是单 agent 的十几倍，只把它放在真正需要并行的环节，账才算得平。

检索由管线完成（见 pipeline/common.py），本 agent 只负责把检索到的片段
整理成**带来源的证据**。分工是刻意的：**检索是工具，不是模型能力。**
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from arksol.agents.base import Agent
from arksol.state import RunState


class EvidenceAgent(Agent):
    name = "evidence"
    tag = "evidence"
    schema_name = "EvidenceBundle"
    max_tokens = 1000

    def __init__(self, provider, *, topic: str, topic_instruction: str, snippets: Sequence[Dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(provider, **kwargs)
        self.topic = topic
        self._topic_instruction = topic_instruction
        self._snippets = list(snippets)
        # trace 里按主题分开命名，评测时能看出「哪个主题没取到证据」
        self.name = f"evidence:{topic}"

    def system_prompt(self) -> str:
        return (
            "你是大模型平台的技术取证工程师。你只从给定的知识库片段里取信息，"
            "**不允许使用片段之外的知识**，也不允许推测。你只输出 JSON。"
        )

    def instruction(self) -> str:
        return (
            f"本轮的取证主题是「{self.topic}」：{self._topic_instruction}\n\n"
            "下方 JSON 里的 snippets 是检索到的知识库片段。请把它们整理成证据条目：\n"
            "- 每条证据必须有 text（陈述事实）与 source_id（片段里给的来源编号）\n"
            "- confidence 直接沿用片段给的可信度\n"
            "- 片段里没有、但该主题确实需要的信息，写进 missing（这比编一条证据有价值得多）\n"
            "- 如果片段为空，items 留空数组，并在 missing 里说明\n\n"
            "注意：条目 id 由管线统一分配，你按顺序给 EV-001、EV-002 即可。"
        )

    def payload(self, state: RunState) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "topic_instruction": self._topic_instruction,
            "snippets": self._snippets,
        }
