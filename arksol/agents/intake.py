"""Intake · 需求理解（单线程）。

它的产出决定了后面所有环节。这里最重要的一个设计是：
**信息不足时先追问，不要硬编方案。**

理由是有实证的：把完整指令拆成多轮逐步给出，正确率会明显低于一次性给全 ——
因为模型会在早期轮次就假设前提、过早固化方案，之后补充的信息很难推翻它。
所以「问清楚」在这个项目里不是客套，是**一道有产出的工序**。
"""

from __future__ import annotations

from typing import Any, Dict

from arksol.agents.base import Agent
from arksol.state import RunState


class IntakeAgent(Agent):
    name = "intake"
    tag = "intake"
    schema_name = "Requirement"
    max_tokens = 1000

    def system_prompt(self) -> str:
        return (
            "你是大模型平台的产品解决方案分析师，负责把客户的口语化诉求翻译成结构化需求。"
            "你只输出 JSON，不写任何解释。"
        )

    def instruction(self) -> str:
        return (
            "把客户原话整理成结构化需求，字段含义如下：\n"
            "- business_scenario：客户到底要解决什么业务问题（一句话）\n"
            "- capabilities：需要模型提供的能力（数组）\n"
            "- constraints：客户明确说出的约束，kind 只能取 "
            "cost / latency / compliance / concurrency / quality / other，"
            "critical 表示「这条不满足方案就不成立」\n"
            "- open_questions：还没问清楚、且**答案会改变方案**的问题；每条必须写 why_critical\n"
            "- sufficient：信息是否足以出方案\n\n"
            "判断规则：如果客户没说清关键约束、或表述含糊到无法选型，"
            "把 sufficient 设为 false 并给出 open_questions —— 此时不要猜、不要替客户假设。"
        )

    def payload(self, state: RunState) -> Dict[str, Any]:
        # context 的键用英文 snake_case：它既是 mock provider 读取的结构化镜像，
        # 也是真实模型看到的 JSON 块 —— 与契约字段同名，少一层翻译就少一处出错。
        return {"user_input": state.user_input}
