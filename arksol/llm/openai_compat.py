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
import uuid
from typing import Any, Dict, Mapping, Optional

from arksol.env import effective_environ
from arksol.errors import ProviderError
from arksol.llm.base import LLMRequest, LLMResponse, Provider
from arksol.tools.cost import cost_cny, estimate_tokens

# 火山方舟数据面地址。注意：它**已经包含 /api/v3**，
# 用 OpenAI SDK 接入时不要再加 /v1 —— 这是最常见的 404 来源。
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "doubao-seed-2-1-turbo-260628"

_ENV_KEY_NAMES = ("ARK_API_KEY", "OPENCODE_GO_API_KEY", "OPENCODE_API_KEY", "OPENAI_API_KEY")
_ENV_BASE_URL_NAMES = ("ARK_BASE_URL", "OPENCODE_BASE_URL", "OPENAI_BASE_URL")
_ENV_MODEL_NAMES = ("ARK_MODEL", "OPENCODE_MODEL", "OPENAI_MODEL")

# 这两个默认值都是踩过的坑，不是随手写的：
#
# 1. User-Agent：Python urllib 默认发 `Python-urllib/3.10`，会被 Cloudflare 按签名拦掉
#    （实测：HTTP 403 + `error code: 1010`）。所以这里显式声明一个产品名。
# 2. 会话归属头：有些网关（实测 opencode.ai/zen/go）要求请求带上会话 id，
#    缺了会返回 400 `MissingSessionID`。这类要求不适合硬编码进通用 provider，
#    所以做成「按域名给默认值 + 可用环境变量覆盖」。
DEFAULT_USER_AGENT = "ark-solution-lab/0.1.0 (+https://github.com/)"
_SESSION_HEADER_BY_HOST = {"opencode.ai": "x-opencode-session"}


def detect_session_header(base_url: str) -> Optional[str]:
    for host, header in _SESSION_HEADER_BY_HOST.items():
        if host in (base_url or "").lower():
            return header
    return None


class OpenAICompatProvider(Provider):
    name = "openai-compat"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        price: Optional[Mapping[str, float]] = None,
        user_agent: str = DEFAULT_USER_AGENT,
        session_header: Optional[str] = None,
        session_id: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        body_defaults: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not api_key or not str(api_key).strip():
            raise ProviderError(
                "缺少 API Key。请先设置环境变量（例如 ARK_API_KEY），"
                "或者把它写进项目根目录的 .env（该文件已在 .gitignore 里）。"
            )
        self._api_key = str(api_key).strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._price = dict(price) if price else None

        self._user_agent = user_agent or DEFAULT_USER_AGENT
        # 网关要求的会话归属头：显式给了就用给的，否则按域名推断
        self._session_header = session_header or detect_session_header(self.base_url)
        self._session_id = session_id or uuid.uuid4().hex
        self._extra_headers: Dict[str, str] = dict(extra_headers or {})
        if self._session_header:
            self._extra_headers.setdefault(self._session_header, self._session_id)
        self._body_defaults: Dict[str, Any] = dict(body_defaults or {})

    def _request_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": self._user_agent,
        }
        headers.update(self._extra_headers)
        return headers

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

        resolved_timeout = float(environ.get("ARK_TIMEOUT") or timeout)
        extra_headers: Dict[str, str] = {}
        raw_headers = environ.get("ARK_EXTRA_HEADERS")
        if raw_headers:
            try:
                extra_headers = {str(k): str(v) for k, v in json.loads(raw_headers).items()}
            except (json.JSONDecodeError, AttributeError):
                raise ProviderError("ARK_EXTRA_HEADERS 必须是 JSON 对象，例如 {\"X-Foo\": \"bar\"}") from None

        body_defaults: Dict[str, Any] = {}
        raw_body = environ.get("ARK_EXTRA_BODY")
        if raw_body:
            try:
                body_defaults = dict(json.loads(raw_body))
            except (json.JSONDecodeError, TypeError):
                raise ProviderError("ARK_EXTRA_BODY 必须是 JSON 对象") from None
        # 推理型模型会把 max_tokens 先花在思考上，常常导致 content 为空。
        # 做对照实验时我们关心的是「管线结构」，所以默认建议关掉思考 ——
        # 三条流水线一起关，比较才公平。
        if str(environ.get("ARK_DISABLE_THINKING", "")).strip().lower() in ("1", "true", "yes", "on"):
            body_defaults.setdefault("thinking", {"type": "disabled"})

        return cls(
            api_key=resolved_key or "",
            model=resolved_model,
            base_url=resolved_base,
            timeout=resolved_timeout,
            price=price,
            user_agent=environ.get("ARK_USER_AGENT") or DEFAULT_USER_AGENT,
            session_header=environ.get("ARK_SESSION_HEADER") or None,
            extra_headers=extra_headers,
            body_defaults=body_defaults,
        )

    # -- 调用 ---------------------------------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        # 顺序很重要：先铺厂商默认参数，再铺本次请求的额外参数，
        # 最后写 model / messages 等核心字段 —— 核心字段不允许被覆盖。
        payload: Dict[str, Any] = dict(self._body_defaults)
        payload.update(request.extra_body)
        payload.update(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.user},
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
        )
        if request.stop:
            payload["stop"] = request.stop

        http_request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._request_headers(),
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
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        finish_reason = choice.get("finish_reason")

        if not text.strip():
            # 推理型模型会把额度先花在思考上。这里给一个**能定位问题**的报错，
            # 而不是让它退化成一次说不清的契约失败。
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            hint = "（模型只产出了思考内容，content 为空 —— 请调大 max_tokens 或关闭思考模式）" if reasoning else ""
            raise ProviderError(
                f"模型返回了空 content{hint}。finish_reason={finish_reason}，"
                f"reasoning_chars={len(str(reasoning))}"
            )

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
