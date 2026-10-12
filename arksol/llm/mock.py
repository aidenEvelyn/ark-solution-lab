"""确定性 mock 提供方。

它**不是「假的模型」**，而是一段规则脚本，用途只有一个：把管线机制跑通并可回归测试 ——
契约校验、本地重试、预算熔断、trace 落盘、审校退回、故障注入。

⚠️ 边界必须说清（README 与结果表里都要标）：

- mock 下跑出来的**机制指标**（契约合规率、重试次数、熔断、成本记账）是有效的；
- mock 下跑出来的**质量指标**（无依据断言率、约束覆盖率）反映的是**这段脚本的设定**，
  **不能当作模型能力证据**。要得到有解释力的质量数字，必须用真实模型跑一遍。

故障注入（fault injection）是这里最有价值的部分：它让我们能在**不花一分钱**的前提下，
验证「契约层和 Critic 到底拦不拦得住坏输出」。
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from arksol.llm.base import LLMRequest, LLMResponse, Provider
from arksol.tools.cost import cost_cny, estimate_tokens
from arksol.verify import NO_EVIDENCE_MARK

FAULT_KINDS = (
    "malformed_json",  # 吐一段非 JSON 的道歉
    "empty",  # 什么都不吐
    "missing_field",  # 合法 JSON 但缺必填字段
    "unsupported_claim",  # 伪造一条引用（指向不存在的证据）
    "unknown_source",  # 证据挂一个未登记的来源
    "constraint_ignored",  # 方案对客户约束只字不提
)

# 「欠规格」的信号词。刻意**不含**「大概 / 可能」这类词 ——
# 「QPS 大概 50」是正常的工程表达，不是需求不清楚。
# 这个取舍本身就是一个产品判断：误判成「信息不足」会让客户多答一轮问题，
# 误判成「信息充足」会让方案建立在猜测上。本项目选择前者，但把判据写死、可复现。
_VAGUE_HINTS = ("还没想好", "不确定", "不清楚", "先看看", "随便", "你看着办")

_CONSTRAINT_LABEL = {
    "compliance": "数据合规要求",
    "latency": "延迟要求",
    "concurrency": "并发要求",
    "cost": "成本要求",
    "quality": "质量要求",
    "other": "其他要求",
}


# --------------------------------------------------------------------------------------
# 故障注入
# --------------------------------------------------------------------------------------


@dataclass
class Fault:
    """在第 `at_call` 次（含之后 `times` 次）指定 tag 的调用上注入一个故障。"""

    kind: str
    tag: Optional[str] = None
    at_call: int = 1
    times: int = 1

    def __post_init__(self) -> None:
        if self.kind not in FAULT_KINDS:
            raise ValueError(f"未知的故障类型 {self.kind!r}，可选：{FAULT_KINDS}")

    def matches(self, tag: str, call_index: int) -> bool:
        if self.tag is not None and self.tag != tag:
            return False
        return self.at_call <= call_index < self.at_call + self.times


def _load(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def apply_fault(kind: str, text: str, request: LLMRequest) -> str:
    """把一次正常输出改造成「坏输出」。改不动就原样返回（说明这个故障不适用于该 tag）。"""
    if kind == "malformed_json":
        return "抱歉，我无法按要求输出 JSON 格式的结果。"
    if kind == "empty":
        return ""

    data = _load(text)
    if not isinstance(data, dict):
        return text

    if kind == "missing_field":
        for key in ("followup_questions", "summary", "open_questions", "items"):
            if key in data:
                del data[key]
                break
    elif kind == "unsupported_claim":
        if isinstance(data.get("claims"), list):
            data["claims"].append(
                {
                    "text": "该模型支持 512k 上下文窗口，可一次处理整本书。",
                    "evidence_ref": "EV-999",
                    "kind": "limit",
                }
            )
    elif kind == "unknown_source":
        items = data.get("items")
        if isinstance(items, list) and items:
            items[0]["source_id"] = "S99"
    elif kind == "constraint_ignored":
        data["summary"] = "整体方案可行，建议尽快启动 PoC。"
        data["deployment"] = "公有云在线推理"
        data["claims"] = [
            {"text": "模型能力可以覆盖主要业务场景。", "evidence_ref": NO_EVIDENCE_MARK, "kind": "capability"}
        ]
    return _dump(data)


# --------------------------------------------------------------------------------------
# 各环节的默认脚本
# --------------------------------------------------------------------------------------

_HANDLERS: Dict[str, Callable[[LLMRequest], str]] = {}


def handler(tag: str):
    def decorate(fn: Callable[[LLMRequest], str]) -> Callable[[LLMRequest], str]:
        _HANDLERS[tag] = fn
        return fn

    return decorate


def _contains(text: str, words: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(word.lower() in lowered for word in words)


def _short(text: Any, limit: int = 60) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return "（无内容）"
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def _claim_kind(text: str) -> str:
    if _contains(text, ("元", "价", "计费", "成本", "费用")):
        return "price"
    if _contains(text, ("上下文", "窗口", "支持", "工具调用", "接口")):
        return "limit"
    if _contains(text, ("报错", "404", "401", "429", "限流", "排查")):
        return "performance"
    if _contains(text, ("内网", "合规", "私有化", "数据安全")):
        return "compliance"
    return "capability"


def _extract_constraints(text: str) -> List[Dict[str, Any]]:
    constraints: List[Dict[str, Any]] = []
    if _contains(text, ("内网", "合规", "私有化", "不能出", "不出网", "数据安全")):
        constraints.append({"kind": "compliance", "detail": "数据不能出内网", "critical": True})
    if _contains(text, ("延迟", "实时", "响应", "毫秒", "秒内")):
        constraints.append({"kind": "latency", "detail": "对响应延迟有要求", "critical": False})
    if _contains(text, ("并发", "qps", "同时", "峰值")):
        constraints.append({"kind": "concurrency", "detail": "存在并发压力", "critical": False})
    if _contains(text, ("预算", "成本", "费用", "便宜", "多少钱")):
        constraints.append({"kind": "cost", "detail": "有成本约束", "critical": True})
    return constraints


def _extract_capabilities(text: str) -> List[str]:
    capabilities = ["大模型在线推理"]
    if _contains(text, ("问答", "知识库", "客服", "检索")):
        capabilities.append("知识库问答")
    if _contains(text, ("审核", "筛查", "合规检查")):
        capabilities.append("内容审核")
    if _contains(text, ("生成", "文案", "报告", "摘要", "改写")):
        capabilities.append("内容生成")
    if _contains(text, ("图片", "扫描", "拍照", "发票", "识别")):
        capabilities.append("图片理解")
    if _contains(text, ("代码", "研发", "编程")):
        capabilities.append("代码辅助")
    return capabilities


@handler("intake")
def _h_intake(request: LLMRequest) -> str:
    raw = str(request.context.get("user_input") or request.user)
    text = raw.strip()

    constraints = _extract_constraints(text)
    # 判定「欠规格」的规则是刻意写死的、可复现的：
    # 太短 / 含模糊词 / 一条约束都没说出来 → 认为信息不足，先追问再动手
    underspecified = len(text) < 14 or _contains(text, _VAGUE_HINTS) or not constraints

    open_questions: List[Dict[str, Any]] = []
    if underspecified:
        open_questions = [
            {
                "question": "这个能力最终给谁用、每天大概多少量？",
                "why_critical": "决定用在线推理还是批量推理，直接影响成本结构",
            },
            {
                "question": "数据能不能出内网？",
                "why_critical": "这一条决定公有云还是私有化，是整个方案的第一个分叉",
            },
        ]

    return _dump(
        {
            "business_scenario": _short(text, 40),
            "capabilities": _extract_capabilities(text),
            "constraints": constraints,
            "open_questions": open_questions,
            "sufficient": not underspecified,
        }
    )


@handler("evidence")
def _h_evidence(request: LLMRequest) -> str:
    topic = str(request.context.get("topic") or "取证")
    snippets = request.context.get("snippets") or []

    items: List[Dict[str, Any]] = []
    for index, snippet in enumerate(snippets[:4], start=1):
        items.append(
            {
                "id": f"EV-{index:03d}",
                "text": _short(snippet.get("text", ""), 80),
                "source_id": str(snippet.get("source_id") or "S01"),
                "confidence": str(snippet.get("confidence") or "中"),
            }
        )

    missing = [] if items else [f"{topic}：知识库里没有检索到相关条目"]
    return _dump({"topic": topic, "items": items, "missing": missing})


def _writer_payload(context: Mapping[str, Any]) -> Dict[str, Any]:
    requirement = context.get("requirement") or {}
    items = list(context.get("evidence_items") or [])
    scenario = str(requirement.get("business_scenario") or "客户场景")
    constraints = list(requirement.get("constraints") or [])

    has_compliance = any(c.get("kind") == "compliance" for c in constraints)
    has_cost = any(c.get("kind") == "cost" for c in constraints)

    model_choice = "doubao-seed-2-1-turbo（轻量旗舰，成本与延迟更优）" if has_cost else "doubao-seed-2-1-pro（旗舰）"
    deployment = "私有化 / 专属部署" if has_compliance else "公有云在线推理"

    claims: List[Dict[str, Any]] = []
    for item in items[:3]:
        claims.append(
            {
                "text": _short(item.get("text", ""), 60),
                "evidence_ref": str(item.get("id") or NO_EVIDENCE_MARK),
                "kind": _claim_kind(str(item.get("text", ""))),
            }
        )
    if not claims:
        claims.append(
            {
                "text": f"关于「{scenario}」目前没有检索到可引用的资料，以下结论基于通用经验。",
                "evidence_ref": NO_EVIDENCE_MARK,
                "kind": "capability",
            }
        )

    labels = "、".join(_CONSTRAINT_LABEL.get(str(c.get("kind")), "其他要求") for c in constraints) or "客户未明确约束"
    summary = (
        f"针对「{scenario}」，建议采用 {model_choice}，部署方式为{deployment}；"
        f"方案已就客户提出的{labels}逐条对齐。"
    )

    return {
        "summary": summary,
        "model_choice": model_choice,
        "deployment": deployment,
        "claims": claims,
        "followup_questions": [],
    }


@handler("writer")
def _h_writer(request: LLMRequest) -> str:
    return _dump(_writer_payload(request.context))


@handler("single")
def _h_single(request: LLMRequest) -> str:
    """基线 A：一次调用、不看知识库 —— 断言只能标 EV-NONE。"""
    requirement = request.context.get("requirement") or {}
    scenario = str(requirement.get("business_scenario") or "客户场景")
    constraints = list(requirement.get("constraints") or [])
    labels = "、".join(_CONSTRAINT_LABEL.get(str(c.get("kind")), "其他要求") for c in constraints) or "客户未明确约束"

    return _dump(
        {
            "summary": (
                f"针对「{scenario}」，建议使用旗舰大模型配合提示词工程实现，预计 2–4 周可上线；"
                f"方案会关注客户提出的{labels}。"
            ),
            "model_choice": "doubao-seed-2-1-pro（旗舰）",
            "deployment": "公有云在线推理",
            "claims": [
                {
                    "text": "该系列模型支持 256k 上下文窗口，可覆盖多数企业知识库场景。",
                    "evidence_ref": NO_EVIDENCE_MARK,
                    "kind": "limit",
                },
                {
                    "text": "该模型输入约 6 元/百万 tokens、输出约 30 元/百万 tokens。",
                    "evidence_ref": NO_EVIDENCE_MARK,
                    "kind": "price",
                },
                {
                    "text": "平台提供知识库与插件能力，可满足常见集成需求。",
                    "evidence_ref": NO_EVIDENCE_MARK,
                    "kind": "capability",
                },
            ],
            "followup_questions": [],
        }
    )


@handler("critic")
def _h_critic(request: LLMRequest) -> str:
    """mock 的 Critic 不额外提问题 —— 因为确定性的规则层已经把能判的都判完了。

    真实模型下这一层才会补「软」问题（措辞不切题、承诺过度、没回应客户的真实意图）。
    这本身就是一条值得写进结论的发现：**审校的价值主要来自确定性规则，而不是让模型再读一遍。**
    """
    draft = request.context.get("draft") or {}
    claims = draft.get("claims") or []
    return _dump({"verdict": "pass", "issues": [], "checked_claims": len(claims)})


# --------------------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------------------


class MockProvider(Provider):
    name = "mock"

    def __init__(
        self,
        *,
        model: str = "mock-deterministic-v1",
        faults: Iterable[Union[Fault, Mapping[str, Any]]] = (),
        latency_ms: float = 0.0,
        price: Optional[Mapping[str, float]] = None,
        handlers: Optional[Mapping[str, Callable[[LLMRequest], str]]] = None,
    ) -> None:
        self.model = model
        self.faults = [f if isinstance(f, Fault) else Fault(**dict(f)) for f in faults]
        self._latency_ms = float(latency_ms)
        self._price = dict(price) if price else None
        self._handlers: Dict[str, Callable[[LLMRequest], str]] = dict(_HANDLERS)
        if handlers:
            self._handlers.update(handlers)
        self._calls: Dict[str, int] = {}
        # 取证环节是多线程调用的，计数器必须加锁，否则故障注入的 at_call 会飘
        self._lock = threading.Lock()

    @property
    def call_counts(self) -> Dict[str, int]:
        return dict(self._calls)

    @property
    def total_calls(self) -> int:
        return sum(self._calls.values())

    def complete(self, request: LLMRequest) -> LLMResponse:
        tag = request.tag or "default"
        with self._lock:
            self._calls[tag] = self._calls.get(tag, 0) + 1
            call_index = self._calls[tag]

        fn = self._handlers.get(tag)
        # 未知 tag 返回一个合法但不符合任何契约的空对象 —— 让契约层去报错，
        # 而不是在这里悄悄兜底成一个看起来正常的输出。
        text = fn(request) if fn else "{}"

        fault_kind: Optional[str] = None
        for fault in self.faults:
            if fault.matches(tag, call_index):
                text = apply_fault(fault.kind, text, request)
                fault_kind = fault.kind
                break

        prompt_tokens = estimate_tokens(request.system) + estimate_tokens(request.user)
        completion_tokens = estimate_tokens(text)
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
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=self._latency_ms,
            cost_cny=cost,
            fault=fault_kind,
            usage_is_estimated=True,
        )
