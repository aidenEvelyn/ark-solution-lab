"""从 raw.json 重新渲染报告，**不必重跑模型**。

为什么需要它：真实模型的评测要跑很久、还要花钱。改一次报告的措辞就重跑一遍全量，
是对时间和额度的浪费。只要 `raw.json` 里留着完整的原始记录，报告就是可以随时重算的视图。

这也顺带说明一件事：**报告的每个数字都能回到 raw.json 与 traces/ 里复核**，
不是跑完就散掉的输出。
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Tuple

from arksol.errors import ArksolError
from arksol.evaluation.report import render_report
from arksol.evaluation.runner import CaseRun, FaultRow, PipelineSummary


def _rehydrate(cls: Any, data: Dict[str, Any]) -> Any:
    """按 dataclass 的字段名单过滤后重建对象。

    `to_dict()` 会额外塞进派生指标（比率、均值），那些不是字段，构造时要滤掉。
    """
    names = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in names})


def load_raw(path: Path | str) -> Tuple[Dict[str, Any], List[PipelineSummary], List[CaseRun], List[FaultRow]]:
    raw_path = Path(path)
    if not raw_path.exists():
        raise ArksolError(f"找不到原始记录 {raw_path}")
    payload = json.loads(raw_path.read_text(encoding="utf-8"))

    meta = payload.get("meta", {})
    summaries = [_rehydrate(PipelineSummary, item) for item in payload.get("summaries", [])]
    runs = [_rehydrate(CaseRun, item) for item in payload.get("runs", [])]
    faults = [_rehydrate(FaultRow, item) for item in payload.get("fault_rows", [])]
    return meta, summaries, runs, faults


def recompute_derived(summaries: List[PipelineSummary], runs: List[CaseRun]) -> List[PipelineSummary]:
    """从逐条记录重算「按用例聚合」的字段。

    这样即使 raw.json 是更早版本生成的（那时还没有这两个字段），报告也能正确渲染 ——
    派生的东西应该能从原始记录重算，而不是只能跟着跑一遍。
    """
    by_pipeline = {summary.pipeline: summary for summary in summaries}
    for summary in summaries:
        summary.draft_followup_count = 0
        summary.failed_cases = []
    for run in runs:
        summary = by_pipeline.get(run.pipeline)
        if summary is None:
            continue
        if (run.metrics or {}).get("followup_asked"):
            summary.draft_followup_count += 1
        if not run.ok:
            summary.failed_cases.append(run.case_id)
    return summaries


def rebuild_report(raw_path: Path | str, out_path: Path | str | None = None) -> str:
    meta, summaries, runs, faults = load_raw(raw_path)
    summaries = recompute_derived(summaries, runs)
    text = render_report(summaries=summaries, fault_rows=faults, meta=meta, runs=runs)

    target = Path(out_path) if out_path is not None else Path(raw_path).parent / "report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    return str(target)
