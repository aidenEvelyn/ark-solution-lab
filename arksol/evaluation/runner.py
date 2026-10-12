"""对照实验执行器。

两轮实验：

1. **正常路径**：20 条用例 × 3 条流水线，看调用次数、token、成本、无依据断言、约束覆盖。
2. **故障注入**：故意让写作环节产出坏输出（伪造引用 / 忽略约束 / 非法 JSON），
   看**坏输出能不能流到最终产物里**。

第二轮才是这个仓库最有价值的部分：它把「审校有没有用」从形容词变成数字，
而且**同一批坏输出会分别喂给有审校和没审校的流水线** —— 这是唯一能把功劳归因清楚的做法。
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from arksol.budget import Budget
from arksol.errors import ArksolError
from arksol.evaluation.cases import Case, split_by_followup
from arksol.llm import Fault, MockProvider
from arksol.pipeline import Harness, RunRecord, run_pipeline
from arksol.trace import Tracer, read_trace

ProviderFactory = Callable[[], Any]


# --------------------------------------------------------------------------------------
# 正常路径
# --------------------------------------------------------------------------------------


@dataclass
class CaseRun:
    case_id: str
    pipeline: str
    ok: bool
    needs_followup: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    rounds: int = 0
    evidence_count: int = 0
    critique_verdict: Optional[str] = None
    error: Optional[str] = None
    stop_reason: Optional[str] = None
    contract_errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineSummary:
    pipeline: str
    cases: int = 0
    ok: int = 0
    failed: int = 0
    claims_total: int = 0
    claims_unsupported: int = 0
    claims_dangling: int = 0
    judgeable: int = 0
    missed: int = 0
    followup_cases: int = 0
    followup_hit: int = 0
    enough_cases: int = 0
    over_ask: int = 0
    calls: int = 0
    tokens: int = 0
    cost_cny: float = 0.0
    latency_ms: float = 0.0
    contract_errors: int = 0
    rewrites: int = 0
    critic_runs: int = 0
    critic_pass: int = 0
    intake_judged: int = 0
    intake_sufficient_count: int = 0

    # -- 派生指标 -----------------------------------------------------------------------

    @property
    def success_rate(self) -> float:
        return self.ok / self.cases if self.cases else 0.0

    @property
    def intake_sufficient_rate(self) -> float:
        return self.intake_sufficient_count / self.intake_judged if self.intake_judged else 0.0

    @property
    def unsupported_rate(self) -> float:
        return self.claims_unsupported / self.claims_total if self.claims_total else 0.0

    @property
    def coverage_rate(self) -> float:
        return (self.judgeable - self.missed) / self.judgeable if self.judgeable else 1.0

    @property
    def followup_hit_rate(self) -> float:
        return self.followup_hit / self.followup_cases if self.followup_cases else 1.0

    @property
    def over_ask_rate(self) -> float:
        return self.over_ask / self.enough_cases if self.enough_cases else 0.0

    @property
    def avg_calls(self) -> float:
        return self.calls / self.cases if self.cases else 0.0

    @property
    def avg_tokens(self) -> float:
        return self.tokens / self.cases if self.cases else 0.0

    @property
    def avg_cost_cny(self) -> float:
        return self.cost_cny / self.cases if self.cases else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms / self.cases if self.cases else 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "success_rate": round(self.success_rate, 4),
                "unsupported_rate": round(self.unsupported_rate, 4),
                "coverage_rate": round(self.coverage_rate, 4),
                "followup_hit_rate": round(self.followup_hit_rate, 4),
                "over_ask_rate": round(self.over_ask_rate, 4),
                "intake_sufficient_rate": round(self.intake_sufficient_rate, 4),
                "avg_calls": round(self.avg_calls, 2),
                "avg_tokens": round(self.avg_tokens, 1),
                "avg_cost_cny": round(self.avg_cost_cny, 6),
                "avg_latency_ms": round(self.avg_latency_ms, 1),
            }
        )
        return data


def _run_one(
    case: Case,
    pipeline: str,
    provider_factory: ProviderFactory,
    *,
    trace_path: Optional[Path],
    trace_run_id: str,
    budget: Optional[Budget] = None,
    force_draft: bool = False,
) -> RunRecord:
    provider = provider_factory()
    tracer = Tracer(trace_path, run_id=trace_run_id) if trace_path is not None else Tracer(None)
    try:
        harness = Harness.build(provider, tracer=tracer, budget=budget or Budget(), force_draft=force_draft)
        return run_pipeline(pipeline, case.input, harness=harness, case_id=case.id)
    finally:
        tracer.close()


def _count_trace_events(trace_path: Path, event: str) -> Dict[str, int]:
    """按 run_id 统计某种事件出现了几次（用于统计契约失败）。"""
    counts: Dict[str, int] = {}
    if not trace_path.exists():
        return counts
    for record in read_trace(trace_path):
        if record.get("event") == event:
            run_id = str(record.get("run_id", "?"))
            counts[run_id] = counts.get(run_id, 0) + 1
    return counts


def run_matrix(
    cases: Sequence[Case],
    pipelines: Sequence[str],
    provider_factory: ProviderFactory,
    *,
    trace_dir: Optional[Path] = None,
    budget: Optional[Budget] = None,
    force_draft: bool = False,
) -> tuple[List[CaseRun], List[PipelineSummary]]:
    """正常路径的对照实验。每条流水线的 trace 写进一个文件，按 case id 分 run_id。"""
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)

    runs: List[CaseRun] = []
    summaries: Dict[str, PipelineSummary] = {name: PipelineSummary(pipeline=name) for name in pipelines}

    for pipeline in pipelines:
        trace_path = (trace_dir / f"{pipeline}.jsonl") if trace_dir is not None else None
        for case in cases:
            record = _run_one(
                case,
                pipeline,
                provider_factory,
                trace_path=trace_path,
                trace_run_id=case.id,
                budget=budget,
                force_draft=force_draft,
            )
            metrics = record.metrics or {}
            usage = record.usage or {}
            critique = record.critique or {}

            run = CaseRun(
                case_id=case.id,
                pipeline=pipeline,
                ok=record.ok,
                needs_followup=case.needs_followup,
                metrics=metrics,
                usage=usage,
                elapsed_ms=record.elapsed_ms,
                rounds=record.rounds,
                evidence_count=record.evidence_count,
                critique_verdict=critique.get("verdict"),
                error=record.error,
                stop_reason=record.stop_reason,
            )
            runs.append(run)

            summary = summaries[pipeline]
            summary.cases += 1
            summary.ok += 1 if record.ok else 0
            summary.failed += 0 if record.ok else 1
            summary.claims_total += int(metrics.get("claims_total", 0))
            summary.claims_unsupported += int(metrics.get("claims_unsupported", 0))
            summary.claims_dangling += int(metrics.get("claims_dangling", 0))
            summary.judgeable += int(metrics.get("constraints_judgeable", 0))
            summary.missed += int(metrics.get("constraints_missed", 0))
            summary.calls += int(usage.get("calls", 0))
            summary.tokens += int(usage.get("total_tokens", 0))
            summary.cost_cny += float(usage.get("cost_cny", 0.0))
            summary.latency_ms += float(record.elapsed_ms)
            summary.rewrites += max(0, record.rounds - 1)

            asked = bool(metrics.get("followup_asked"))
            if case.needs_followup:
                summary.followup_cases += 1
                summary.followup_hit += 1 if asked else 0
            else:
                summary.enough_cases += 1
                summary.over_ask += 1 if asked else 0

            intake = metrics.get("intake_sufficient")
            if intake is not None:
                summary.intake_judged += 1
                summary.intake_sufficient_count += 1 if intake else 0

            if critique:
                summary.critic_runs += 1
                summary.critic_pass += 1 if critique.get("verdict") == "pass" else 0

        # 契约失败次数来自 trace，而不是模型自报
        if trace_path is not None:
            contract_errors = _count_trace_events(trace_path, "contract_error")
            summaries[pipeline].contract_errors = sum(contract_errors.values())
            for run in runs:
                if run.pipeline == pipeline:
                    run.contract_errors = contract_errors.get(run.case_id, 0)

    return runs, [summaries[name] for name in pipelines]


# --------------------------------------------------------------------------------------
# 故障注入
# --------------------------------------------------------------------------------------

FAULT_SCENARIOS: tuple[Dict[str, Any], ...] = (
    {
        "kind": "unsupported_claim",
        "label": "伪造引用（偶发：只坏第一次）",
        "times": 1,
        "desc": "写作环节凭空造了一条指向不存在证据的断言。偶发故障，重试/退回有救。",
    },
    {
        "kind": "constraint_ignored",
        "label": "忽略客户约束（持续：一直坏）",
        "times": 9,
        "desc": "写作环节始终不回应客户约束。持续故障，退回了也改不动 —— 审校不是万能的。",
    },
    {
        "kind": "malformed_json",
        "label": "输出不是合法 JSON（偶发）",
        "times": 1,
        "desc": "最基础的契约失败，靠本地重试吸收。",
    },
)


@dataclass
class FaultRow:
    scenario: str
    pipeline: str
    runs: int = 0
    caught: int = 0
    leaked: int = 0
    failed: int = 0
    rounds_total: int = 0

    @property
    def caught_rate(self) -> float:
        return self.caught / self.runs if self.runs else 0.0

    @property
    def leaked_rate(self) -> float:
        return self.leaked / self.runs if self.runs else 0.0

    @property
    def avg_rounds(self) -> float:
        return self.rounds_total / self.runs if self.runs else 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "caught_rate": round(self.caught_rate, 4),
                "leaked_rate": round(self.leaked_rate, 4),
                "avg_rounds": round(self.avg_rounds, 2),
            }
        )
        return data


def run_fault_experiment(
    cases: Sequence[Case],
    pipelines: Sequence[str],
    *,
    trace_dir: Optional[Path] = None,
    limit: int = 6,
    scenarios: Sequence[Dict[str, Any]] = FAULT_SCENARIOS,
    force_draft: bool = False,
) -> List[FaultRow]:
    """同一批坏输出，分别喂给有审校和没审校的流水线。"""
    subset = [case for case in cases if not case.needs_followup][:limit]
    if not subset:
        raise ArksolError("故障注入实验需要至少一条信息充足的用例")

    rows: List[FaultRow] = []
    for scenario in scenarios:
        kind = str(scenario["kind"])
        times = int(scenario.get("times", 1))
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)

        for pipeline in pipelines:
            row = FaultRow(scenario=str(scenario["label"]), pipeline=pipeline)
            trace_path = (trace_dir / f"fault-{kind}-{pipeline}.jsonl") if trace_dir is not None else None

            for case in subset:
                record = _run_one(
                    case,
                    pipeline,
                    lambda: MockProvider(faults=[Fault(kind=kind, tag="writer", at_call=1, times=times)]),
                    trace_path=trace_path,
                    trace_run_id=case.id,
                    force_draft=force_draft,
                )
                metrics = record.metrics or {}
                critique = record.critique or {}

                row.runs += 1
                row.rounds_total += max(1, record.rounds)

                caught = bool(critique.get("verdict") == "revise") or record.rounds > 1
                row.caught += 1 if caught else 0

                leaked = (
                    (not record.ok)
                    or int(metrics.get("claims_dangling", 0)) > 0
                    or int(metrics.get("claims_unsupported", 0)) > 0
                    or int(metrics.get("constraints_missed", 0)) > 0
                )
                row.leaked += 1 if leaked else 0
                row.failed += 0 if record.ok else 1

            rows.append(row)

    return rows


# --------------------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------------------


def write_csv(path: Path, runs: Sequence[CaseRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "case_id",
        "pipeline",
        "ok",
        "needs_followup",
        "claims_total",
        "claims_unsupported",
        "claims_dangling",
        "unsupported_rate",
        "constraints_judgeable",
        "constraints_missed",
        "coverage_rate",
        "followup_asked",
        "calls",
        "total_tokens",
        "cost_cny",
        "elapsed_ms",
        "rounds",
        "critique_verdict",
        "contract_errors",
        "error",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for run in runs:
        row = {
            "case_id": run.case_id,
            "pipeline": run.pipeline,
            "ok": run.ok,
            "needs_followup": run.needs_followup,
            "calls": run.usage.get("calls"),
            "total_tokens": run.usage.get("total_tokens"),
            "cost_cny": run.usage.get("cost_cny"),
            "elapsed_ms": run.elapsed_ms,
            "rounds": run.rounds,
            "critique_verdict": run.critique_verdict,
            "contract_errors": run.contract_errors,
            "error": run.error,
            **run.metrics,
        }
        writer.writerow(row)
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")


def write_raw(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def utc_stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
