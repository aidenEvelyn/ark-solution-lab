"""arksol 命令行入口。

当前为骨架：``--help`` / ``--version`` 可用，子命令在后续步骤实现。
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from arksol import __version__

_NOT_YET = "[arksol] 该子命令尚未实现（见 README 的施工状态）。"
_PIPELINES = ("single", "single-guarded", "multi")
_PROVIDERS = ("mock", "openai-compat")


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
    demo.add_argument("--provider", default="mock", choices=_PROVIDERS)
    demo.add_argument("--case", default=None, help="用例 id；缺省跑第一条")
    demo.add_argument("--pipeline", default="multi", choices=_PIPELINES)

    run = sub.add_parser("run", help="对一条自定义输入跑指定流水线")
    run.add_argument("--pipeline", default="multi", choices=_PIPELINES)
    run.add_argument("--provider", default="mock", choices=_PROVIDERS)
    run.add_argument("--input", default=None, help="客户原话；缺省从标准输入读")

    ev = sub.add_parser("eval", help="跑对照实验并生成报告")
    ev.add_argument("--provider", default="mock", choices=_PROVIDERS)
    ev.add_argument("--pipelines", default=",".join(_PIPELINES))
    ev.add_argument("--cases", default="eval/cases.jsonl")
    ev.add_argument("--out", default="results/")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    print(f"[arksol {__version__}] command={args.command}", file=sys.stderr)
    print(_NOT_YET)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
