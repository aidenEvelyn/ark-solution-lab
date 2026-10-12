"""Critic · 独立审校。

两层，顺序是刻意的：

1. **规则层（确定性）**：引用完整性、约束覆盖 —— 这些答案本来就在手上，
   让模型再读一遍既贵又不稳。**能在代码里判定的，就不消耗模型调用。**
2. **模型层**：只有在规则层通过之后才调用，用来补「软」问题
   （承诺了没依据的上线时间、措辞与客户真实意图不符等）。

按本项目的 mock 设定，模型层通常抓不到额外问题 —— 这本身就是一条值得写进结论的发现：
**审校带来的价值主要来自确定性规则，而不是让模型再读一遍。**

为什么必须独立于 Writer：让同一个上下文既生产又检查，等于没检查。
Critic 拿到的是**结构化的产出物**（claims 数组），因此能做机械核对，而不是凭感觉打分。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from arksol.agents.base import Agent
from arksol.state import RunState
from arksol.verify import review


class CriticAgent(Agent):
    name = "critic"
    tag = "critic"
    schema_name = "Critique"
    max_tokens = 2000

    def __init__(
        self,
        provider,
        *,
        evidence_items: Sequence[Dict[str, Any]] = (),
        requirement: Optional[Dict[str, Any]] = None,
        keyword_map: Optional[Dict[str, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(provider, **kwargs)
        self._evidence_items = list(evidence_items)
        self._evidence_ids = {str(item.get("id", "")) for item in self._evidence_items}
        self._requirement = requirement or {}
        self._keyword_map = keyword_map

    def system_prompt(self) -> str:
        return (
            "你是独立审校。你的任务不是夸奖方案写得好，而是找出**它在什么地方没有依据、"
            "或者回避了客户的约束**。你只输出 JSON。"
        )

    def instruction(self) -> str:
        return (
            "下方 JSON 里的 draft 是待审校的方案草稿。"
            "确定性规则已经检查过引用完整性与约束覆盖，你只需要补充规则抓不到的「软」问题：\n"
            "- 承诺了没有依据的时间、数量或效果\n"
            "- 答非所问、回避了客户的真实意图\n"
            "- 结论与客户约束存在隐含冲突\n\n"
            "没有发现问题时，verdict 输出 pass、issues 输出空数组 —— 不要为了显得有用而凑问题。\n"
            "issues[].type 只能取 unsupported_claim / uncovered_constraint / other。"
        )

    def payload(self, state: RunState) -> Dict[str, Any]:
        return {
            "draft": state.draft or {},
            "evidence_items": [
                {"id": item.get("id"), "text": item.get("text"), "source_id": item.get("source_id")}
                for item in self._evidence_items
            ],
            "constraints": self._requirement.get("constraints", []),
        }

    def run(self, state: RunState) -> Dict[str, Any]:
        rule_result = review(
            state.draft or {},
            evidence_ids=self._evidence_ids,
            requirement=self._requirement,
            keyword_map=self._keyword_map,
        )

        if rule_result["verdict"] == "revise":
            # 规则层已经判负，省掉一次模型调用
            self._trace(
                event="rule_reject",
                payload={"issues": [issue["type"] for issue in rule_result["issues"]]},
                level="warn",
            )
            return rule_result

        model_result = super().run(state)
        issues: List[Dict[str, Any]] = list(rule_result["issues"]) + list(model_result.get("issues") or [])
        return {
            "verdict": "pass" if not issues else "revise",
            "issues": issues,
            "checked_claims": max(
                int(rule_result["checked_claims"]), int(model_result.get("checked_claims") or 0)
            ),
        }
