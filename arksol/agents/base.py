"""Agent 基类。

在这个项目里，一个 agent 就是三样东西：**提示词 + 契约 + 重试策略**。
机器部分（调用、记账、trace、契约校验、本地重试）全部由基类完成 —— 这样四个角色之间的
差异是可见、可比较的，而不是散落在四份长得差不多的代码里。

一条关键纪律：**契约不通过就在本地重试，绝不带病往下一步传。**
多智能体最大的一类失败是「规格问题」，而链条越长，模糊规格被反复解释、误差在交接处放大。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

from arksol.budget import BudgetGuard, Meter
from arksol.contracts import parse_contract, render_schema
from arksol.errors import ContractError
from arksol.llm.base import LLMRequest, Provider, render_user_prompt
from arksol.state import RunState
from arksol.trace import Tracer


class Agent:
    """所有角色的基类。子类只需要实现三个方法。"""

    name: str = "agent"
    tag: str = "agent"
    schema_name: str = ""
    max_retries: int = 2
    temperature: float = 0.0
    max_tokens: int = 1200

    def __init__(
        self,
        provider: Provider,
        *,
        tracer: Optional[Tracer] = None,
        guard: Optional[BudgetGuard] = None,
        meter: Optional[Meter] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self._provider = provider
        self._tracer = tracer
        self._guard = guard
        self._meter = meter
        if max_retries is not None:
            self.max_retries = max_retries

    # -- 子类实现 -----------------------------------------------------------------------

    def system_prompt(self) -> str:
        raise NotImplementedError

    def instruction(self) -> str:
        raise NotImplementedError

    def payload(self, state: RunState) -> Dict[str, Any]:
        raise NotImplementedError

    # -- 机器部分 -----------------------------------------------------------------------

    def build_request(self, state: RunState) -> LLMRequest:
        payload = self.payload(state)
        return LLMRequest(
            system=self.system_prompt() + self._schema_block(),
            user=render_user_prompt(self.instruction(), payload),
            tag=self.tag,
            schema_name=self.schema_name,
            # context 是「这段提示词从哪些结构化数据渲染出来的」：
            # 真实 provider 忽略它，mock provider 直接读它，从而产出被证据支撑的输出。
            context=payload,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _schema_block(self) -> str:
        """把契约原文塞进系统提示词。

        没有这一段，模型只能靠散文描述去猜字段名 —— 而猜错之后的重试是收敛不了的。
        """
        if not self.schema_name:
            return ""
        return (
            "\n\n【输出格式 · 必须严格遵守】\n"
            "你必须输出**且仅输出**一个 JSON 对象，不要写任何解释文字或 Markdown 围栏。\n"
            "字段名、层级、必填项必须与下面的 JSON Schema 完全一致 —— "
            "不要改名、不要增删字段；标注了 enum 的字段只能取列出的值。\n"
            "```json\n" + render_schema(self.schema_name) + "\n```"
        )

    def _trace(self, **kwargs: Any) -> None:
        if self._tracer is not None:
            self._tracer.emit(agent=self.name, **kwargs)

    def run(self, state: RunState) -> Dict[str, Any]:
        """调用 → 记账 → trace → 契约校验 → 不合格则本地重试。"""
        request = self.build_request(state)
        last_error: Optional[ContractError] = None

        for attempt in range(1, self.max_retries + 2):
            if self._guard is not None:
                self._guard.step()

            response = self._provider.complete(request)

            if self._meter is not None:
                self._meter.add(self.name, response)
            if self._guard is not None:
                self._guard.charge(tokens=response.total_tokens, cost_cny=response.cost_cny)

            self._trace(
                event="call",
                payload={"attempt": attempt, "fault": response.fault, "schema": self.schema_name},
                usage=response.usage_dict(),
                elapsed_ms=response.latency_ms,
            )

            try:
                data = parse_contract(response.text, self.schema_name)
            except ContractError as exc:
                last_error = exc
                state.record_attempt(stage=self.name, round=attempt, outcome="contract_error", reason=str(exc))
                # **必须把模型的原始输出记下来**：否则「它为什么不符合契约」永远查不出来，
                # 只能看到一个笼统的解析失败。截断是为了别让 trace 无限膨胀。
                self._trace(
                    event="contract_error",
                    payload={
                        "attempt": attempt,
                        "error": str(exc),
                        "raw_response": response.text[:4000],
                        "raw_chars": len(response.text),
                    },
                    level="warn",
                )
                # 把失败原因写回提示词再试一次 —— 这是「本地重试」而不是「带病传递」
                request = replace(
                    request,
                    user=request.user
                    + f"\n\n【上一次的输出不符合契约，必须修正】{exc}\n"
                    + "请只输出符合契约的 JSON，不要任何解释文字，不要代码围栏之外的任何内容。",
                )
                continue

            state.record_attempt(stage=self.name, round=attempt, outcome="ok")
            self._trace(event="contract_ok", payload={"attempt": attempt})
            return data

        assert last_error is not None
        raise last_error
