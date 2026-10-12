"""LLM 提供方的公共抽象。

设计上只有两个概念：一次请求、一次响应。刻意不引入「消息历史」——
本项目的状态放在外部（见 state.py），每个环节只拿它需要的那一份，
这和把整段历史塞进上下文的做法是相反的。

`LLMRequest.context` 是这段提示词**从哪些结构化数据渲染出来的**：
真实 provider 忽略它（信息已经在 prompt 里），mock provider 直接读它。
这样 mock 就能在不需要模型的情况下，产出被证据支撑的、可复现的输出。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LLMRequest:
    system: str
    user: str
    tag: str = ""
    schema_name: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.0
    max_tokens: int = 1200
    stop: Optional[list] = None
    # 厂商特有的请求体字段（例如关闭思考模式：{"thinking": {"type": "disabled"}}）。
    # 放在这里而不是硬编码进 provider —— 各家网关的参数名不一样，别把差异写死在一处。
    extra_body: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_cny: float = 0.0
    fault: Optional[str] = None
    usage_is_estimated: bool = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def usage_dict(self) -> Dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}


class Provider(ABC):
    """所有提供方的接口。实现方必须保证：**没有返回 usage 时也要给出估计值**。"""

    name: str = "provider"
    model: str = "unknown"

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover - 接口
        raise NotImplementedError


def render_user_prompt(instruction: str, payload: Dict[str, Any]) -> str:
    """把结构化数据渲染成提示词里的 JSON 块。

    用 json.dumps 而不是自然语言拼接：模型读结构化输入比读散文更稳，
    而且出问题时能一眼看出喂进去了什么。
    """
    import json

    body = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return f"{instruction}\n\n```json\n{body}\n```"
