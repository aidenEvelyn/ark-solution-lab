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
    # 真实模型下，一份带若干约束与追问项的 JSON 需要更多额度；
    # 额度不够会表现为「契约解析失败」，别把它误读成模型不听话。
    max_tokens = 2400

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
            "- sufficient：信息是否足以**出一版方案草稿**\n\n"
            "关键：sufficient 判断的是「够不够出草稿」，不是「够不够定稿」。"
            "售前第一次交流本来就拿不到全部细节，把标准定成「够定稿」会导致一份方案都出不来。\n\n"
            "把缺失信息分成两级，分开处理：\n"
            "A. **阻断级** —— 缺了它，方案的方向就无从谈起。例如：不知道客户要解决什么业务问题；"
            "不知道数据能不能出内网（这决定公有云还是私有化，是第一个分叉）；完全没有量级与成本信息。\n"
            "   → sufficient 设为 **false**，把问题写进 open_questions，此时不要出方案。\n"
            "B. **细化级** —— 缺了它方案会有偏差，但方向不变。例如：文档规模、延迟目标、"
            "准确率期望、预算是否包含硬件。\n"
            "   → sufficient 设为 **true**，正常出方案；同时把这些点写进 open_questions，"
            "它们会随方案一起交给客户确认。\n\n"
            "可操作的判据：**如果一个问题的答案无论是什么，都不会改变你推荐的模型档位和部署方式，"
            "那它就是细化级，不要用它把方案拦下来。**"
        )

    def payload(self, state: RunState) -> Dict[str, Any]:
        # context 的键用英文 snake_case：它既是 mock provider 读取的结构化镜像，
        # 也是真实模型看到的 JSON 块 —— 与契约字段同名，少一层翻译就少一处出错。
        return {"user_input": state.user_input}
