"""三条对照流水线。

| 名字 | 做什么 | 想回答的问题 |
|---|---|---|
| `no-evidence` | Intake → 一次调用出方案（**不看知识库**） | 最省事的做法差在哪 |
| `evidence-once` | Intake → 检索 → 一次调用出方案（看知识库，但无审校） | **便宜的改良能不能顶替多智能体** |
| `multi` | Intake → 并行取证 → 写 → 独立审校（不合格退回，最多 2 轮） | 多花的钱买到了什么 |

三条共享同一个 Intake —— 「理解需求」不是本次实验的变量。这样：
**A→B 隔离出「有没有证据」，B→C 隔离出「有没有编排与审校」。**
一次只改一个变量，结论才解释得清。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from arksol.agents.critic import CriticAgent
from arksol.agents.intake import IntakeAgent
from arksol.agents.writer import MemoryPlanAgent, WriterAgent
from arksol.budget import Meter
from arksol.contracts import validate_named
from arksol.errors import ContractError, ProviderError, RepeatActionDetected, BudgetExceeded
from arksol.pipeline.common import (
    EVIDENCE_TOPICS,
    Harness,
    RunRecord,
    finish_record,
    gather_evidence,
    normalize_evidence,
    snippets_to_evidence,
)
from arksol.state import RunState

PIPELINE_NAMES = ("no-evidence", "evidence-once", "multi")

# 最多退回改写 2 次（即最多 3 次尝试）。到线即停并如实标注「未通过审校」。
MAX_REWRITES = 2

_RECOVERABLE = (ContractError, BudgetExceeded, RepeatActionDetected, ProviderError)


def _agent_kwargs(harness: Harness, meter: Meter) -> Dict[str, Any]:
    return {"tracer": harness.tracer, "guard": harness.guard, "meter": meter}


def _new_record(harness: Harness, pipeline: str, client_input: str, case_id: Optional[str]) -> RunRecord:
    return RunRecord(
        pipeline=pipeline,
        provider=getattr(harness.provider, "name", "unknown"),
        client_input=client_input,
        case_id=case_id,
    )


def build_followup(requirement: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """信息不足 → 组装追问清单（不额外调用模型，直接复用 Intake 的产出）。"""
    if requirement.get("sufficient", True):
        return None
    questions = list(requirement.get("open_questions") or [])
    if not questions:
        return None
    followup = {"questions": questions, "assumptions": []}
    validate_named(followup, "FollowUp")
    return followup


def _intake(harness: Harness, state: RunState, meter: Meter, record: RunRecord) -> Optional[RunRecord]:
    """跑 Intake。

    返回一个填好的 record 表示「到此结束」；返回 None 表示继续往下走。
    `harness.force_draft` 为 True 时，即使判为信息不足也继续 —— 但判断照样记录在 metrics 里。

    欠规格还有一个分支要区分清楚：Intake 说「信息不足」却**一条追问都没给**。
    那不是「它想追问」，那是产出不可用 —— 如实报错，不替它编问题。
    """
    requirement = IntakeAgent(harness.provider, **_agent_kwargs(harness, meter)).run(state)
    state.requirement = requirement
    record.requirement = requirement

    sufficient = bool(requirement.get("sufficient", True))
    followup = build_followup(requirement)

    if sufficient:
        return None

    if followup is None:
        record.error = "intake 判定信息不足，但没有给出任何追问项（契约允许，但结果不可用）"
        state.stop_reason = record.error
        return record

    if harness.force_draft:
        # 强制出方案模式：不在这里结束，但把「它本来想追问」记进指标，单独报告
        return None

    state.followup = followup
    record.followup = followup
    record.ok = True
    return record


def _clip(text: Any, limit: int = 200) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


# --------------------------------------------------------------------------------------


def run_no_evidence(
    client_input: str, *, harness: Harness, case_id: Optional[str] = None
) -> RunRecord:
    meter = Meter()
    state = RunState(user_input=client_input)
    record = _new_record(harness, "no-evidence", client_input, case_id)
    started = time.perf_counter()

    try:
        if _intake(harness, state, meter, record) is None:
            draft = MemoryPlanAgent(
                harness.provider, requirement=record.requirement or {}, **_agent_kwargs(harness, meter)
            ).run(state)
            state.draft = draft
            record.draft = draft
            record.ok = True
    except _RECOVERABLE as exc:
        record.error = f"{type(exc).__name__}: {exc}"
        state.stop_reason = record.error

    return finish_record(record, state, meter, started)


def run_evidence_once(
    client_input: str, *, harness: Harness, case_id: Optional[str] = None
) -> RunRecord:
    """基线 B：检索一次，塞进提示词，一次成型 —— 没有审校、没有退回。"""
    meter = Meter()
    state = RunState(user_input=client_input)
    record = _new_record(harness, "evidence-once", client_input, case_id)
    started = time.perf_counter()

    try:
        if _intake(harness, state, meter, record) is None:
            query = " ".join(topic["query"] for topic in EVIDENCE_TOPICS)
            snippets = harness.retrieve(query, top_k=5, topic="综合检索")
            items = snippets_to_evidence(snippets, topic="综合检索")
            state.evidence = [{"topic": "综合检索", "items": items, "missing": []}]
            record.missing_evidence = []

            draft = WriterAgent(
                harness.provider,
                requirement=record.requirement or {},
                evidence_items=items,
                **_agent_kwargs(harness, meter),
            ).run(state)
            state.draft = draft
            record.draft = draft
            record.ok = True
    except _RECOVERABLE as exc:
        record.error = f"{type(exc).__name__}: {exc}"
        state.stop_reason = record.error

    return finish_record(record, state, meter, started)


def run_multi(client_input: str, *, harness: Harness, case_id: Optional[str] = None) -> RunRecord:
    """C：并行只读取证 → 单线程撰写 → 独立审校（不合格退回，最多 2 轮）。"""
    meter = Meter()
    state = RunState(user_input=client_input)
    record = _new_record(harness, "multi", client_input, case_id)
    started = time.perf_counter()

    try:
        if _intake(harness, state, meter, record) is None:
            bundles = gather_evidence(harness, state, meter)
            normalized, items = normalize_evidence(bundles)
            state.evidence = normalized
            record.missing_evidence = [
                str(note) for bundle in normalized for note in (bundle.get("missing") or [])
            ]

            revision_notes: Optional[List[Dict[str, Any]]] = None
            previous_signature: Optional[tuple] = None
            previous_issue_count: Optional[int] = None

            for attempt in range(1, MAX_REWRITES + 2):
                draft = WriterAgent(
                    harness.provider,
                    requirement=record.requirement or {},
                    evidence_items=items,
                    revision_notes=revision_notes,
                    **_agent_kwargs(harness, meter),
                ).run(state)
                state.draft = draft
                record.rounds = attempt

                critique = CriticAgent(
                    harness.provider,
                    evidence_items=items,
                    requirement=record.requirement or {},
                    **_agent_kwargs(harness, meter),
                ).run(state)
                state.critiques.append(critique)

                if critique.get("verdict") == "pass":
                    break

                issues = list(critique.get("issues") or [])
                signature = tuple(sorted(str(issue.get("detail", ""))[:80] for issue in issues))

                # 停止条件一：连续两轮被**同一批问题**退回 —— 改不动了。
                if signature and signature == previous_signature:
                    state.stop_reason = f"连续两轮被同一批问题退回（{len(signature)} 条），判定为改不动，提前停止"
                    break

                # 停止条件二：**改写没有收敛**（问题数没有减少）。
                # 真实模型下常见这种情形：模型层审校每轮都能提出新的、也确实成立的意见，
                # 但那些意见多半源于「客户信息不足」而不是「写作质量差」—— 改到第几轮都收不了。
                # 判据用「问题数是否下降」而不是「签名是否相同」，因为每轮的措辞都在变。
                if previous_issue_count is not None and len(issues) >= previous_issue_count:
                    state.stop_reason = (
                        f"改写未收敛（问题数 {previous_issue_count} → {len(issues)}，没有减少），"
                        "提前停止 —— 这类问题多半要靠客户补信息，不是靠再改一遍"
                    )
                    break

                previous_signature = signature
                previous_issue_count = len(issues)

                if attempt == MAX_REWRITES + 1:
                    state.stop_reason = f"审校连续 {attempt} 轮未通过，已停止并如实标注"
                    break

                revision_notes = issues

            record.draft = state.draft
            record.critique = state.critiques[-1] if state.critiques else None
            record.ok = True
    except _RECOVERABLE as exc:
        record.error = f"{type(exc).__name__}: {exc}"
        state.stop_reason = record.error

    return finish_record(record, state, meter, started)


_RUNNERS = {
    "no-evidence": run_no_evidence,
    "evidence-once": run_evidence_once,
    "multi": run_multi,
}


def run_pipeline(
    name: str, client_input: str, *, harness: Harness, case_id: Optional[str] = None
) -> RunRecord:
    runner = _RUNNERS.get(name)
    if runner is None:
        raise ValueError(f"未知的流水线 {name!r}，可选：{PIPELINE_NAMES}")
    return runner(client_input, harness=harness, case_id=case_id)
