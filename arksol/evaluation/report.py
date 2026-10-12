"""把实验结果渲染成 Markdown 报告。

报告的第一节不是数据，而是**这份数据在什么条件下才成立** ——
一个不写清适用范围的数字表，比没有数字更危险。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from arksol.evaluation.runner import FAULT_SCENARIOS, FaultRow, PipelineSummary
from arksol.tools.cost import format_cny


def _pct(value: float) -> str:
    return f"{value:.0%}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _by_name(summaries: Sequence[PipelineSummary]) -> Dict[str, PipelineSummary]:
    return {summary.pipeline: summary for summary in summaries}


def _conclusions(summaries: Sequence[PipelineSummary], fault_rows: Sequence[FaultRow]) -> List[str]:
    """结论从数字里长出来，不写死 —— 换一批用例、换个模型，这里会跟着变。"""
    by_name = _by_name(summaries)
    lines: List[str] = []

    base = by_name.get("no-evidence")
    once = by_name.get("evidence-once")
    multi = by_name.get("multi")

    if base and once:
        if abs(once.avg_calls - base.avg_calls) < 0.05:
            cost_phrase = f"而调用次数一次都没多（都是 {base.avg_calls:.1f} 次），只是提示词变长了"
        else:
            cost_phrase = (
                f"而代价只是平均调用次数从 {base.avg_calls:.1f} 次变成 {once.avg_calls:.1f} 次"
            )
        lines.append(
            f"- **给证据比不给证据重要得多。** 不检索时无依据断言率 {_pct(base.unsupported_rate)}"
            f"（{base.claims_unsupported}/{base.claims_total} 条）；检索一次之后降到 "
            f"{_pct(once.unsupported_rate)}（{once.claims_unsupported}/{once.claims_total} 条），"
            f"{cost_phrase}。"
        )
    if once and multi and once.avg_tokens:
        ratio = multi.avg_tokens / once.avg_tokens
        lines.append(
            f"- **多智能体的开销是实打实的。** 同样一批用例，`multi` 平均 {multi.avg_tokens:.0f} tokens/次，"
            f"`evidence-once` 是 {once.avg_tokens:.0f}，倍率 **{ratio:.1f}×**"
            f"（平均调用 {multi.avg_calls:.1f} 次 vs {once.avg_calls:.1f} 次）。"
        )
        if abs(multi.unsupported_rate - once.unsupported_rate) < 1e-9:
            lines.append(
                "- **在正常路径上，多出来的开销没有换来质量提升。** 两条流水线的无依据断言率相同"
                "（都是 0）—— 这条结论和 Anthropic / Cognition 的工程复盘一致："
                "编排本身不产生正确性，它只在**故障路径**上产生价值。"
            )
        else:
            lines.append(
                f"- 在正常路径上，`multi` 的无依据断言率 {_pct(multi.unsupported_rate)}，"
                f"`evidence-once` 是 {_pct(once.unsupported_rate)} —— 差异见上表。"
            )

    if fault_rows:
        no_critic = [row for row in fault_rows if row.pipeline != "multi"]
        with_critic = [row for row in fault_rows if row.pipeline == "multi"]
        if no_critic and with_critic:
            leak_without = sum(row.leaked for row in no_critic)
            total_without = sum(row.runs for row in no_critic)
            leak_with = sum(row.leaked for row in with_critic)
            total_with = sum(row.runs for row in with_critic)
            lines.append(
                f"- **审校的价值体现在故障路径上。** 注入同样的坏输出：没有审校的流水线有 "
                f"**{leak_without}/{total_without}** 次把问题带进了最终产物；"
                f"有审校的是 **{leak_with}/{total_with}** 次。"
            )

        persistent = [row for row in fault_rows if "持续" in row.scenario]
        if persistent:
            details = "，".join(
                f"`{row.pipeline}` {row.leaked}/{row.runs}"
                for row in sorted(persistent, key=lambda item: item.pipeline)
            )
            if any(row.leaked for row in persistent):
                lines.append(
                    f"- **审校不是万能的。** 对「一直坏」的模型，问题最终**都**留在了产物里（{details}）——"
                    "审校能发现并退回，但改不动一个持续失效的写作环节。"
                    "**这种情况下正确的动作是停下来找人或者换模型，而不是让它再试一次。**"
                )

    return lines or ["- 本次运行没有产生可比较的结论（检查用例与流水线配置）。"]


def _intake_agreement(runs: Sequence[Any]) -> Dict[str, Any] | None:
    """比对 Intake 的「信息是否充足」判断与用例的人工标注。"""
    per_case: Dict[str, Any] = {}
    for run in runs:
        if run.case_id in per_case:
            continue
        judged = (run.metrics or {}).get("intake_sufficient")
        if judged is None:
            continue
        per_case[run.case_id] = bool(judged)
    if not per_case:
        return None

    label_by_case = {run.case_id: bool(run.needs_followup) for run in runs}
    mismatches = []
    for case_id, judged in per_case.items():
        label_needs_followup = label_by_case.get(case_id, False)
        if label_needs_followup == (not judged):  # 一致：标注说要追问 == 模型判不足
            continue
        mismatches.append((case_id, label_needs_followup, judged))

    return {"total": len(per_case), "agree": len(per_case) - len(mismatches), "mismatches": mismatches}


def render_report(
    *,
    summaries: Sequence[PipelineSummary],
    fault_rows: Sequence[FaultRow],
    meta: Dict[str, Any],
    runs: Sequence[Any] = (),
) -> str:
    by_name = _by_name(summaries)
    base = by_name.get("no-evidence")
    once = by_name.get("evidence-once")

    parts: List[str] = []
    parts.append("# 对照实验结果")
    parts.append("")
    parts.append(
        f"> 生成时间：{meta.get('generated_at', '?')} ｜ provider：`{meta.get('provider', '?')}`"
        f" ｜ 模型：`{meta.get('model', '?')}`"
    )
    parts.append(
        f"> 用例：{meta.get('cases', 0)} 条（其中 {meta.get('underspecified_cases', 0)} 条为欠规格）"
        f" ｜ 用例文件：`{meta.get('cases_file', 'evaluation/cases.jsonl')}`"
        f" ｜ 流水线：{', '.join(f'`{name}`' for name in meta.get('pipelines', []))}"
    )
    parts.append(f"> 复现：`{meta.get('command', 'python -m arksol eval --provider mock')}`")
    if meta.get("force_draft"):
        parts.append(
            "> ⚠️ 本次启用了 `--force-draft`：**即使 Intake 判为信息不足也继续出方案**。"
            "目的是把「方案质量对照」与「追问行为」两件事拆开测 —— "
            "追问判定照样被记录，见第五节。"
        )
    parts.append("")

    # ---- 适用范围：必须先说清楚 -------------------------------------------------------
    parts.append("## 一、先读这段：哪些数字有效，哪些没有")
    parts.append("")
    if meta.get("provider") == "mock":
        parts.append(
            "本次运行用的是 `mock` 提供方 —— 它不是模型，而是一段**确定性规则脚本**。"
            "因此："
        )
        parts.append("")
        parts.append(
            _table(
                ["数字", "在 mock 下是否有效", "为什么"],
                [
                    ("调用次数 / token / 成本", "✅ 有效", "prompt 是管线真实拼出来的，token 计数基于真实文本"),
                    ("契约合规率 / 重试次数 / 熔断", "✅ 有效", "这是管线机制，与生成内容的模型无关"),
                    ("故障注入下的拦截率", "✅ 有效", "坏输出由脚本精确注入，可复现"),
                    ("无依据断言率 / 约束覆盖率", "⚠️ 反映脚本设定", "mock 的产出由写死的规则决定，不是模型行为"),
                    ("追问命中率", "⚠️ 自证", "mock 的判定规则与用例标注共用同一套关键词，必然一致"),
                    ("延迟", "❌ 无意义", "mock 不产生真实网络与推理耗时"),
                ],
            )
        )
        parts.append("")
        parts.append(
            "**结论：这份报告能证明「管线机制是通的、账是算得清的」，"
            "不能证明「某个模型有多强」。** 要得到后者，请用真实模型重跑（见文末命令）。"
        )
    else:
        parts.append(
            f"本次运行用的是真实模型 `{meta.get('model', '?')}`（provider `{meta.get('provider')}`）。"
            "但请注意以下几点仍然成立："
        )
        parts.append("")
        parts.append(
            "- 用例是**自建的 20 条**，不是行业基准，结论只在这个设定下成立；\n"
            "- 约束覆盖率用**关键词匹配**做代理，措辞不同的同义表达可能漏判；\n"
            "- 「无依据断言率」是幻觉的**代理指标**：一条断言挂着真实来源，仍可能是对来源的误读；\n"
            "- 成本按公开单价估算，未计缓存命中、批量推理折扣与免费额度抵扣（**不是账单**）。"
        )
    parts.append("")

    # ---- 表 1 -------------------------------------------------------------------------
    parts.append("## 二、表 1 · 同一批用例下，三条流水线各花了多少")
    parts.append("")
    rows = []
    for summary in summaries:
        rows.append(
            [
                f"`{summary.pipeline}`",
                _pct(summary.success_rate),
                summary.claims_total,
                f"{summary.claims_unsupported}（{_pct(summary.unsupported_rate)}）",
                f"{_pct(summary.coverage_rate)}",
                f"{summary.avg_calls:.1f}",
                f"{summary.avg_tokens:.0f}",
                format_cny(summary.avg_cost_cny),
                f"{summary.avg_latency_ms:.0f} ms",
            ]
        )
    parts.append(
        _table(
            ["流水线", "成功率", "断言总数", "无依据断言", "约束覆盖", "平均调用", "平均 tokens", "单次成本", "平均耗时"],
            rows,
        )
    )
    parts.append("")

    # ---- 表 2 -------------------------------------------------------------------------
    parts.append("## 三、表 2 · 故障注入：坏输出能不能流到最终产物里")
    parts.append("")
    parts.append(
        "做法：**同一批坏输出**分别喂给有审校和没审校的流水线 —— 这是唯一能把功劳归因清楚的方式。"
        "「被识别」指审校判负或发生了退回重写；「仍流入产物」指最终结果里还带着这个问题。"
    )
    parts.append("")
    fault_rows_sorted = sorted(fault_rows, key=lambda row: (row.scenario, row.pipeline))
    parts.append(
        _table(
            ["注入的故障", "流水线", "运行次数", "被识别", "仍流入产物", "平均轮次"],
            [
                [
                    row.scenario,
                    f"`{row.pipeline}`",
                    row.runs,
                    f"{row.caught}（{_pct(row.caught_rate)}）",
                    f"{row.leaked}（{_pct(row.leaked_rate)}）",
                    f"{row.avg_rounds:.1f}",
                ]
                for row in fault_rows_sorted
            ],
        )
    )
    parts.append("")

    parts.append("这三种故障分别考验管线里的**不同那一层**，所以三行的结论完全不同：")
    parts.append("")
    for scenario in FAULT_SCENARIOS:
        parts.append(f"- **{scenario['label']}** —— {scenario['desc']}")
    parts.append("")

    # ---- 表 3 -------------------------------------------------------------------------
    parts.append("## 四、表 3 · 机制指标（这些数字与用哪个模型无关）")
    parts.append("")
    parts.append(
        _table(
            ["流水线", "契约失败次数", "退回重写次数", "审校通过", "该追问时追问", "不该追问却追问"],
            [
                [
                    f"`{summary.pipeline}`",
                    summary.contract_errors,
                    summary.rewrites,
                    f"{summary.critic_pass}/{summary.critic_runs}" if summary.critic_runs else "—",
                    f"{summary.followup_hit}/{summary.followup_cases}" if summary.followup_cases else "—",
                    f"{summary.over_ask}/{summary.enough_cases}" if summary.enough_cases else "—",
                ]
                for summary in summaries
            ],
        )
    )
    parts.append("")

    # ---- Intake 的追问判定 -------------------------------------------------------------
    agreement = _intake_agreement(runs)
    if agreement:
        parts.append("## 五、Intake 的追问判定 vs 人工标注")
        parts.append("")
        parts.append(
            f"一致 **{agreement['agree']}/{agreement['total']}** 条。"
            "这一栏测的是「Intake 什么时候觉得该先追问」，与人工标注（`evaluation/cases.jsonl` 里的 "
            "`needs_followup`）比对。"
        )
        parts.append("")
        if agreement["mismatches"]:
            parts.append("不一致的用例：")
            parts.append("")
            parts.append(
                _table(
                    ["用例", "人工标注", "Intake 判定", "说明"],
                    [
                        [
                            case_id,
                            "应当先追问" if label else "信息充足、应出方案",
                            "判为信息不足" if not judged else "判为信息充足",
                            "**模型比人工标注更保守**" if not judged and not label else "模型认为可以直接出方案",
                        ]
                        for case_id, label, judged in agreement["mismatches"]
                    ],
                )
            )
            parts.append("")
            parts.append(
                "**这不是「模型错了」，而是两种口径的差异**：人工标注按「有没有场景 + 有没有约束」判，"
                "模型按「能不能据此定下模型档位与部署方式」判，后者更严。"
                "两种判法都能自圆其说，真实售前里也确实靠人拿捏 —— 这里如实并列，不替任何一方辩护。"
            )
        else:
            parts.append("没有任何分歧。")
        if meta.get("provider") == "mock":
            parts.append("")
            parts.append(
                "⚠️ **mock 下这一栏是自证的**：mock 判定信息是否充足用的是与用例标注同一套关键词，"
                "所以必然一致。只有在真实模型下这一栏才有信息量。"
            )
        parts.append("")

    # ---- 结论 -------------------------------------------------------------------------
    parts.append("## 六、结论")
    parts.append("")
    parts.extend(_conclusions(summaries, fault_rows))
    parts.append("")

    # ---- 复现 -------------------------------------------------------------------------
    parts.append("## 七、怎么复现")
    parts.append("")
    parts.append("```bash")
    parts.append("# 离线复现（零依赖、不花钱）")
    parts.append("python -m arksol eval --provider mock --out results/")
    parts.append("")
    parts.append("# 用真实模型重跑（需要自己准备 key，只走环境变量）")
    parts.append("export ARK_API_KEY=...")
    parts.append("export ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3")
    parts.append("export ARK_MODEL=doubao-seed-2-1-turbo-260628")
    parts.append("export ARK_PRICE_IN=3 ARK_PRICE_OUT=15   # 元/百万 tokens，用于成本换算")
    parts.append("python -m arksol eval --provider openai-compat --out results/ark/")
    parts.append("```")
    parts.append("")
    parts.append(
        "逐条明细见同目录 `results.csv`；原始记录见 `raw.json`；"
        "每一步的 trace 见 `traces/`（JSONL，可用 `arksol.trace.read_trace` 读回）。"
    )
    parts.append("")

    return "\n".join(parts)
