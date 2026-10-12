"""OpenAI 兼容提供方 —— 覆盖火山方舟（豆包）以及任何兼容端点。

为什么用标准库 urllib 而不是 requests / openai SDK：
这个仓库的卖点之一是「clone 下来不装任何东西就能跑」。多一个依赖就多一道门槛，
而这里的 HTTP 调用本身只有十几行。

**关于密钥**：只从环境变量读，只在请求头里用，绝不写进代码、日志或 trace。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Mapping, Optional

from arksol.env import effective_environ
from arksol.errors import ProviderError
from arksol.llm.base import LLMRequest, LLMResponse, Provider
from arksol.tools.cost import cost_cny, estimate_tokens

# 火山方舟数据面地址。注意：它**已经包含 /api/v3**，
# 用 OpenAI SDK 接入时不要再加 /v1 —— 这是最常见的 404 来源。
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "doubao-seed-2-1-turbo-260628"

_ENV_KEY_NAMES = ("ARK_API_KEY", "OPENAI_API_KEY")
_ENV_BASE_URL_NAMES = ("ARK_BASE_URL", "OPENAI_BASE_URL")
_ENV_MODEL_NAMES = ("ARK_MODEL", "OPENAI_MODEL")


class OpenAICompatProvider(Provider):
    name = "openai-compat"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        price: Optional[Mapping[str, float]] = None,
    ) -> None:
        if not api_key or not str(api_key).strip():
            raise ProviderError(
                "缺少 API Key。请先设置环境变量（例如 ARK_API_KEY），"
                "并把 key 只放在环境变量或 .env 里 —— 不要写进代码。"
            )
        self._api_key = str(api_key).strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._price = dict(price) if price else None

    # -- 构造 ---------------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        dotenv_path: Optional[str] = None,
    ) -> "OpenAICompatProvider":
        """配置来源优先级：显式参数 > 真实环境变量 > `.env` 文件。

        显式传了 `env` 就只用它（便于测试）；否则用 `有效环境`：
        `.env` 打底、真实环境变量覆盖 —— 这样部署环境里设的值不会被仓库里
        一个忘记删的 .env 盖掉。
        """
        environ = dict(env) if env is not None else effective_environ(dotenv_path)

        def first(names) -> Optional[str]:
            for name in names:
                value = environ.get(name)
                if value:
                    return value
            return None

        resolved_key = api_key or first(_ENV_KEY_NAMES)
        resolved_model = model or first(_ENV_MODEL_NAMES) or DEFAULT_MODEL
        resolved_base = base_url or first(_ENV_BASE_URL_NAMES) or DEFAULT_BASE_URL

        price = None
        price_in, price_out = environ.get("ARK_PRICE_IN"), environ.get("ARK_PRICE_OUT")
        if price_in and price_out:
            price = {"price_in_cny_per_mtok": float(price_in), "price_out_cny_per_mtok": float(price_out)}

        return cls(
            api_key=resolved_key or "",
            model=resolved_model,
            base_url=resolved_base,
            timeout=timeout,
            price=price,
        )

    # -- 调用 ---------------------------------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.stop:
            payload["stop"] = request.stop

        http_request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise ProviderError(
                f"HTTP {exc.code}（{self.base_url}）。常见原因见知识库 errors.json。原始响应：{detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"网络错误（{self.base_url}）：{exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(f"响应不是合法 JSON：{exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        choices = body.get("choices") or []
        if not choices:
            raise ProviderError(f"响应里没有 choices：{str(body)[:300]}")
        text = (choices[0].get("message") or {}).get("content") or ""

        usage = body.get("usage") or {}
        if usage.get("prompt_tokens") is not None:
            prompt_tokens = int(usage["prompt_tokens"])
            completion_tokens = int(usage.get("completion_tokens", 0))
            estimated = False
        else:
            prompt_tokens = estimate_tokens(request.system) + estimate_tokens(request.user)
            completion_tokens = estimate_tokens(text)
            estimated = True

        cost = 0.0
        if self._price:
            cost = cost_cny(
                prompt_tokens,
                completion_tokens,
                price_in_cny_per_mtok=self._price["price_in_cny_per_mtok"],
                price_out_cny_per_mtok=self._price["price_out_cny_per_mtok"],
            )

        return LLMResponse(
            text=text,
            provider=self.name,
            model=body.get("model") or self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_cny=cost,
            usage_is_estimated=estimated,
        )
