"""LLM 提供方。

    provider = get_provider("mock")                       # 零依赖、可复现、可注入故障
    provider = get_provider("openai-compat")              # 读环境变量里的 key
    provider = get_provider("ark", model="doubao-...")    # 同上，名字更直白

真实调用前请先设置环境变量（**不要把 key 写进代码**）：

    ARK_API_KEY=...
    ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
    ARK_MODEL=doubao-seed-2-1-turbo-260628
"""

from __future__ import annotations

from typing import Any

from arksol.errors import ProviderError
from arksol.llm.base import LLMRequest, LLMResponse, Provider, render_user_prompt
from arksol.llm.mock import FAULT_KINDS, Fault, MockProvider
from arksol.llm.openai_compat import DEFAULT_BASE_URL, DEFAULT_MODEL, OpenAICompatProvider

PROVIDER_NAMES = ("mock", "openai-compat", "ark")


def get_provider(name: str = "mock", **kwargs: Any) -> Provider:
    key = (name or "mock").strip().lower()
    if key == "mock":
        return MockProvider(**kwargs)
    if key in ("openai-compat", "openai", "ark"):
        return OpenAICompatProvider.from_env(**kwargs)
    raise ProviderError(f"未知的提供方 {name!r}，可选：{PROVIDER_NAMES}")


__all__ = [
    "Fault",
    "FAULT_KINDS",
    "LLMRequest",
    "LLMResponse",
    "MockProvider",
    "OpenAICompatProvider",
    "PROVIDER_NAMES",
    "Provider",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "get_provider",
    "render_user_prompt",
]
