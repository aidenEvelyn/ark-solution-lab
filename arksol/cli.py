"""arksol 命令行入口。

    python -m arksol demo                                  # 跑一条样例，看完整链路
    python -m arksol run --pipeline multi --input "…"       # 跑自己的输入
    python -m arksol eval                                  # 对照实验（见第 5 步）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

from arksol import __version__
from arksol.budget import Budget
from arksol.errors import ArksolError, ProviderError
from arksol.llm import PROVIDER_NAMES, get_provider
from arksol.pipeline import PIPELINE_NAMES, Harness, RunRecord, run_pipeline
from arksol.tools.cost import format_cny
from arksol.trace import Tracer

DEMO_INPUT = "我们想做个内部知识库问答，给客服团队用，但数据不能出内网，QPS 大概 50，预算每月两千左右。"
DEFAULT_CASES = "eval/cases.jsonl"
DEFAULT_TRACE_DIR = "traces"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arksol",
        description=(
            "方舟售前方案实验室：把客户的一句话需求变成带证据的技术方案草稿，"
            "并用对照实验回答「这个场景该不该上多智能体」。"
        ),
        epilog="示例：python -m arksol demo --provider mock",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"arksol {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    demo = sub.add_parser("demo", help="跑一条样例，打印完整链路与 trace 落盘位置")
    demo.add_argument("--provider", default="mock", choices=PROVIDER_NAMES)
    demo.add_argument("--pipeline", default="multi", choices=PIPELINE_NAMES)
    demo.add_argument("--case", default=None, help="用例 id（从 --cases 指定的文件里取）")
    demo.add_argument("--cases", default=DEFAULT_CASES)
    demo.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)

    run = sub.add_parser("run", help="对一条自定义输入跑指定流水线")
    run.add_argument("--pipeline", default="multi", choices=PIPELINE_NAMES)
    run.add_argument("--provider", default="mock", choices=PROVIDER_NAMES)
    run.add_argument("--input", default=None, help="客户原话；缺省从标准输入读")
    run.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)

    ev = sub.add_parser("eval", help="跑对照实验并生成报告")
    ev.add_argument("--provider", default="mock", choices=PROVIDER_NAMES)
    ev.add_argument("--pipelines", default=",".join(PIPELINE_NAMES))
    ev.add_argument("--cases", default=DEFAULT_CASES)
    ev.add_argument("--out", default="results/")

    return parser


# --------------------------------------------------------------------------------------


def load_case(path: str, case_id: str) -> Tuple[str, str]:
    """从 cases.jsonl 里取一条用例。取不到就明确报错，不猜。"""
    case_path = Path(path)
    if not case_path.exists():
        raise ArksolError(f"找不到用例文件 {case_path}（用 --cases 指定，或先跑一次 eval 生成）")
    for line in case_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(case.get("id")) == case_id:
            return case_id, str(case.get("input", ""))
    raise ArksolError(f"用例文件 {case_path} 里没有 id={case_id!r} 的用例")


def make_harness(provider_name: str, trace_dir: str) -> Harness:
    provider = get_provider(provider_name)
    tracer = Tracer(Path(trace_dir) / f"{datetime.now():%Y%m%d-%H%M%S}-{provider_name}.jsonl")
    return Harness.build(provider, tracer=tracer, budget=Budget())


def print_record(record: RunRecord) -> None:
    status = "成功" if record.ok else "失败"
    print(f"── 流水线 {record.pipeline} ｜ provider {record.provider} ｜ 用时 {record.elapsed_ms:.0f} ms ｜ {status}")
    if record.error:
        print(f"   错误：{record.error}")
    if record.stop_reason:
        print(f"   停止原因：{record.stop_reason}")

    requirement = record.requirement or {}
    if requirement:
        print(f"   需求：{requirement.get('business_scenario', '')}")
        print(f"   信息是否充足：{'是' if requirement.get('sufficient') else '否'}")
        for constraint in requirement.get("constraints", []):
            flag = "（关键）" if constraint.get("critical") else ""
            print(f"     · 约束[{constraint.get('kind')}]{flag}：{constraint.get('detail')}")

    if record.followup:
        print("   追问清单：")
        for question in record.followup.get("questions", []):
            print(f"     ? {question['question']}")
            print(f"       ↳ 为什么关键：{question['why_critical']}")

    if record.draft:
        print(f"   方案：{record.draft['summary']}")
        print(
            f"   选型：{record.draft['model_choice']} ｜ 部署：{record.draft['deployment']}"
            f" ｜ 改写轮次：{record.rounds or 1}"
        )
        print(f"   断言 {len(record.draft['claims'])} 条：")
        for claim in record.draft["claims"]:
            print(f"     - [{claim['evidence_ref']}] {claim['text']}")
        if record.draft.get("followup_questions"):
            print(f"   还需要追问：{record.draft['followup_questions']}")

    if record.critique:
        critique = record.critique
        print(
            f"   审校：{critique['verdict']}"
            f"（检查 {critique['checked_claims']} 条断言，{len(critique['issues'])} 个问题）"
        )
        for issue in critique["issues"]:
            print(f"     ! {issue['type']}：{issue['detail']}")

    metrics = record.metrics
    print(
        f"   指标：无依据断言 {metrics['claims_unsupported']}/{metrics['claims_total']}"
        f" ｜ 约束覆盖 {metrics['coverage_rate']:.0%}（{metrics['constraints_missed']} 条未回应）"
        f" ｜ 该追问时追问：{'是' if metrics['followup_asked'] else '否'}"
    )
    usage = record.usage
    estimated = "，含估算" if usage.get("estimated_calls") else ""
    print(
        f"   花费：{usage['calls']} 次调用 ｜ {usage['total_tokens']} tokens"
        f" ｜ {format_cny(usage['cost_cny'])}{estimated}"
    )


def _cmd_demo(args: argparse.Namespace) -> int:
    client_input, case_id = DEMO_INPUT, None
    if args.case:
        case_id, client_input = load_case(args.cases, args.case)
    harness = make_harness(args.provider, args.trace_dir)
    try:
        record = run_pipeline(args.pipeline, client_input, harness=harness, case_id=case_id)
    finally:
        harness.tracer.close()

    print(f"客户原话：{client_input}\n")
    print_record(record)
    print(f"\ntrace 落盘：{harness.tracer.path}")
    return 0 if record.ok else 1


def _cmd_run(args: argparse.Namespace) -> int:
    client_input = args.input if args.input is not None else sys.stdin.read().strip()
    if not client_input:
        print("[arksol] 没有输入。用 --input 或从标准输入喂一段客户原话。", file=sys.stderr)
        return 2
    harness = make_harness(args.provider, args.trace_dir)
    try:
        record = run_pipeline(args.pipeline, client_input, harness=harness)
    finally:
        harness.tracer.close()

    print(f"客户原话：{client_input}\n")
    print_record(record)
    print(f"\ntrace 落盘：{harness.tracer.path}")
    return 0 if record.ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "eval":
        print("[arksol] eval 子命令尚未实现（见 README 的施工状态）。")
        return 3

    try:
        if args.command == "demo":
            return _cmd_demo(args)
        return _cmd_run(args)
    except ProviderError as exc:
        print(f"[arksol] 模型提供方错误：{exc}", file=sys.stderr)
        return 2
    except ArksolError as exc:
        print(f"[arksol] 运行失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[arksol] 已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
