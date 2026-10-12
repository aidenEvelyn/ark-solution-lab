"""流水线公共件：检索、证据规范化、运行记录、指标计算。

其中有一条容易被忽略但很重要的设计：**证据条的 id 由管线统一分配，不由模型生成。**
模型负责内容，不负责身份 —— 不要相信模型生成的标识符，它在并行场景下会撞车。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from arksol.budget import Budget, BudgetGuard, Meter
from arksol.kb import KnowledgeBase, load_default_kb
from arksol.llm.base import Provider
from arksol.state import RunState
from arksol.tools.retriever import BM25Index, build_index
from arksol.trace import Tracer
from arksol.verify import check_constraints, classify_claims

_CONFIDENCE_RANK = {"高": 0, "中": 1, "低": 2}

# 取证主题是写死的三摊，不是让模型自己决定要分几路。
# 理由：让模型决定「要不要拆、拆几路」是一次额外的不可控判断；
# 而在售前场景里，该查的东西其实是稳定的（能不能做、有没有限制、要多少钱）。
EVIDENCE_TOPICS: tuple[Dict[str, str], ...] = (
    {
        "name": "模型选型",
        "query": "模型选型 上下文 旗舰 轻量 能力",
        "instruction": "找出适合该场景的模型档位、上下文窗口与能力边界",
    },
    {
        "name": "限制与合规",
        "query": "数据合规 内网 私有化 部署 限制 接入方式",
        "instruction": "找出与数据合规、部署方式、接入限制相关的条目",
    },
    {
        "name": "计费与性能",
        "query": "计费 成本 单价 缓存 限流 性能 报错",
        "instruction": "找出计费口径、限流与性能相关的条目",
    },
)


# --------------------------------------------------------------------------------------
# 检索
# --------------------------------------------------------------------------------------


def entry_source(kb: KnowledgeBase, entry: Dict[str, Any]) -> tuple[str, str]:
    """取条目的来源编号与**最保守的可信度**（多来源时取最差的那个）。"""
    sources = kb.source_of(entry)
    if not sources:
        return ("", "低")
    worst = max(sources, key=lambda s: _CONFIDENCE_RANK.get(str(s.get("confidence", "低")), 3))
    return (str(sources[0].get("id", "")), str(worst.get("confidence", "低")))


@dataclass
class Harness:
    """一次运行需要的全部依赖。三条流水线共用同一套，保证可比。"""

    kb: KnowledgeBase
    index: BM25Index
    provider: Provider
    tracer: Optional[Tracer] = None
    guard: Optional[BudgetGuard] = None

    @classmethod
    def build(
        cls,
        provider: Provider,
        *,
        kb: Optional[KnowledgeBase] = None,
        tracer: Optional[Tracer] = None,
        budget: Optional[Budget] = None,
        guard: Optional[BudgetGuard] = None,
    ) -> "Harness":
        knowledge = kb or load_default_kb()
        return cls(
            kb=knowledge,
            index=build_index(knowledge),
            provider=provider,
            tracer=tracer,
            guard=guard or BudgetGuard(budget or Budget()),
        )

    def retrieve(self, query: str, *, top_k: int = 3, topic: str = "综合检索") -> List[Dict[str, Any]]:
        """检索并返回带来源的片段。

        每次检索都过一个动作指纹 —— 同一个查询反复重试会被熔断，
        这是「沉默失败」（日志里没有 error，只是在重复同一个无效动作）的守门人。
        """
        if self.guard is not None:
            self.guard.note_action(f"retrieve:{query}")
        hits = self.index.search(query, top_k=top_k)
        return [self._snippet(hit.doc_id, topic, hit.score) for hit in hits]

    def _snippet(self, doc_id: str, topic: str, score: float) -> Dict[str, Any]:
        entry = self.kb.get(doc_id)
        source_id, confidence = entry_source(self.kb, entry)
        return {
            "entry_id": doc_id,
            "title": entry.get("title", ""),
            "source_id": source_id,
            "confidence": confidence,
            "text": entry.get("text", ""),
            "score": score,
            "topic": topic,
        }


def snippets_to_evidence(snippets: Sequence[Dict[str, Any]], *, topic: str = "综合检索") -> List[Dict[str, Any]]:
    """把检索片段直接转成证据条目（不经过模型），id 按顺序统一分配。"""
    items: List[Dict[str, Any]] = []
    for index, snippet in enumerate(snippets, start=1):
        items.append(
            {
                "id": f"EV-{index:03d}",
                "topic": topic,
                "text": str(snippet.get("text", "")),
                "source_id": str(snippet.get("source_id", "")),
                "confidence": str(snippet.get("confidence", "低")),
            }
        )
    return items


def normalize_evidence(bundles: Sequence[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """把多个取证 agent 的证据包**重新编号**成一个统一的证据表。

    模型各自从 EV-001 开始编号，并行时必然撞车 —— 而 id 撞车会让
    「这条断言挂的是哪条证据」这个判断失效。所以身份由管线统一发。
    """
    normalized: List[Dict[str, Any]] = []
    flat: List[Dict[str, Any]] = []
    counter = 0

    for bundle in bundles:
        new_items: List[Dict[str, Any]] = []
        for item in bundle.get("items", []) or []:
            counter += 1
            new_item = dict(item)
            new_item["id"] = f"EV-{counter:03d}"
            new_items.append(new_item)
            flat.append(new_item)
        normalized.append({**bundle, "items": new_items})

    return normalized, flat


def gather_evidence(
    harness: Harness,
    state: RunState,
    meter: Meter,
    *,
    topics: Iterable[Dict[str, str]] = EVIDENCE_TOPICS,
    top_k: int = 3,
    agent_factory=None,
) -> List[Dict[str, Any]]:
    """并行跑多路只读取证。返回按主题顺序排列的证据包（顺序稳定，便于复现）。"""
    from arksol.agents.evidence import EvidenceAgent

    topic_list = list(topics)
    factory = agent_factory or (
        lambda topic, snippets: EvidenceAgent(
            harness.provider,
            topic=topic["name"],
            topic_instruction=topic["instruction"],
            snippets=snippets,
            tracer=harness.tracer,
            guard=harness.guard,
            meter=meter,
        )
    )

    def one(topic: Dict[str, str]) -> Dict[str, Any]:
        snippets = harness.retrieve(topic["query"], top_k=top_k, topic=topic["name"])
        return factory(topic, snippets).run(state)

    if not topic_list:
        return []
    with ThreadPoolExecutor(max_workers=len(topic_list)) as pool:
        # pool.map 保序：结果顺序与主题顺序一致，复现时不会因为线程调度而变
        return list(pool.map(one, topic_list))


# --------------------------------------------------------------------------------------
# 运行记录与指标
# --------------------------------------------------------------------------------------


@dataclass
class RunRecord:
    pipeline: str
    provider: str
    client_input: str
    case_id: Optional[str] = None
    ok: bool = False
    elapsed_ms: float = 0.0
    rounds: int = 0
    evidence_count: int = 0
    requirement: Optional[Dict[str, Any]] = None
    followup: Optional[Dict[str, Any]] = None
    draft: Optional[Dict[str, Any]] = None
    critique: Optional[Dict[str, Any]] = None
    missing_evidence: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    stop_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_metrics(
    *,
    draft: Optional[Dict[str, Any]],
    requirement: Optional[Dict[str, Any]],
    evidence_ids: Iterable[str],
    followup: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """一次运行的指标。三条流水线用**同一把尺子**。"""
    followed_up = bool(followup) or bool((draft or {}).get("followup_questions"))
    base: Dict[str, Any] = {
        "claims_total": 0,
        "claims_unsupported": 0,
        "claims_dangling": 0,
        "unsupported_rate": 0.0,
        "constraints_judgeable": 0,
        "constraints_missed": 0,
        "coverage_rate": 1.0,
        "followup_asked": followed_up,
    }
    if not draft:
        return base

    claims = classify_claims(draft, evidence_ids)
    coverage = check_constraints(requirement or {}, draft)
    base.update(
        {
            "claims_total": claims["total"],
            "claims_unsupported": claims["unsupported_count"],
            "claims_dangling": len(claims["dangling"]),
            "unsupported_rate": round(claims["unsupported_rate"], 4),
            "constraints_judgeable": coverage["judgeable"],
            "constraints_missed": len(coverage["missed"]),
            "coverage_rate": round(coverage["coverage_rate"], 4),
            "followup_asked": followed_up,
        }
    )
    return base


def finish_record(record: RunRecord, state: RunState, meter: Meter, started: float) -> RunRecord:
    record.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    record.usage = meter.snapshot()
    evidence_ids = {str(item.get("id", "")) for item in state.evidence_items()}
    record.metrics = compute_metrics(
        draft=record.draft,
        requirement=record.requirement,
        evidence_ids=evidence_ids,
        followup=record.followup,
    )
    record.evidence_count = len(state.evidence_items())
    # 停止原因必须带出流水线 —— 否则「为什么没通过」这个信息就丢在 state 里了
    if not record.stop_reason and state.stop_reason:
        record.stop_reason = state.stop_reason
    return record
