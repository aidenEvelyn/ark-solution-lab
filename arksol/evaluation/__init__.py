"""评测台：用例、对照实验执行器、报告渲染。

这个包的存在是为了让「多智能体值不值」变成**可以被复核的数字**，
而不是一句形容词。它同时是 Critic 用的那把尺子（`arksol.verify`）的调用方 ——
**三条流水线用同一把尺子打分**，这是结论能站住的前提。
"""

from arksol.evaluation.cases import Case, default_cases_path, load_cases, split_by_followup
from arksol.evaluation.rebuild import load_raw, rebuild_report, recompute_derived
from arksol.evaluation.report import render_report
from arksol.evaluation.runner import (
    FAULT_SCENARIOS,
    CaseRun,
    FaultRow,
    PipelineSummary,
    run_fault_experiment,
    run_matrix,
    utc_stamp,
    write_csv,
    write_raw,
)

__all__ = [
    "FAULT_SCENARIOS",
    "Case",
    "CaseRun",
    "FaultRow",
    "PipelineSummary",
    "default_cases_path",
    "load_cases",
    "load_raw",
    "rebuild_report",
    "recompute_derived",
    "render_report",
    "run_fault_experiment",
    "run_matrix",
    "split_by_followup",
    "utc_stamp",
    "write_csv",
    "write_raw",
]
