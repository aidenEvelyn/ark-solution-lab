"""独立审计器：不依赖模型，用确定性规则检查一份方案草稿。

它有两个身份，这是刻意的：

1. **Critic 的自检层** —— 能在代码里判定的，就不要去问模型。让模型做
   引用完整性检查既贵又不稳，而且答案本来就在手上。
2. **评测台的裁判** —— 三条流水线用**同一把尺子**打分。如果要回答
   「多智能体到底值不值」，评分本身就不能是黑盒，否则结论没法复核。

⚠️ 已知局限（写在 README 里，也写在这里）：
- 「约束覆盖率」用关键词匹配做代理，措辞不同的同义表达可能漏判；
- 「无依据断言率」是幻觉的**代理指标**，一条断言挂着真实来源，仍可能是对来源的误读；
- 单看比率会被钻空子（把断言删光，比率就变成 0）——所以报告里必须**同时给出断言总数**。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

# 基线 A 用它显式声明「我凭记忆写的，没有证据」。
# 有了这个标记，「没有证据支撑的断言」和「引用了不存在来源的断言」才能分开统计 ——
# 后者更严重，因为它伪装成了有依据。
NO_EVIDENCE_MARK = "EV-NONE"

# 约束种类 → 关键词。可被单个用例覆盖。
DEFAULT_CONSTRAINT_KEYWORDS: Dict[str, List[str]] = {
    "compliance": ["内网", "私有化", "合规", "不出网", "本地部署", "数据安全", "专属"],
    "latency": ["延迟", "响应", "时延", "毫秒", "秒内", "实时"],
    "concurrency": ["并发", "qps", "吞吐", "峰值", "同时"],
    "cost": ["成本", "费用", "预算", "单价", "计费", "元"],
    "quality": ["准确", "效果", "质量", "命中", "召回"],
    "other": [],
}


def draft_text(draft: Mapping[str, Any]) -> str:
    """把草稿摊平成一段可做关键词匹配的文本。"""
    parts: List[str] = [
        str(draft.get("summary", "")),
        str(draft.get("model_choice", "")),
        str(draft.get("deployment", "")),
    ]
    for claim in draft.get("claims", []) or []:
        parts.append(str(claim.get("text", "")))
    for question in draft.get("followup_questions", []) or []:
        parts.append(str(question))
    return "\n".join(p for p in parts if p)


def classify_claims(draft: Mapping[str, Any], evidence_ids: Iterable[str]) -> Dict[str, Any]:
    """把断言按「有没有证据支撑」分三类。

    - supported：引用指向本次运行证据表里真实存在的一条证据
    - no_evidence：显式标了 EV-NONE，即「我凭记忆写的」
    - dangling：引用了证据表里不存在的东西 —— 比没有引用更糟，因为它看起来像有依据
    """
    known: Set[str] = set(evidence_ids)
    supported: List[Dict[str, Any]] = []
    no_evidence: List[Dict[str, Any]] = []
    dangling: List[Dict[str, Any]] = []

    for claim in draft.get("claims", []) or []:
        ref = str(claim.get("evidence_ref", ""))
        if ref in known:
            supported.append(claim)
        elif ref == NO_EVIDENCE_MARK:
            no_evidence.append(claim)
        else:
            dangling.append(claim)

    total = len(supported) + len(no_evidence) + len(dangling)
    unsupported = len(no_evidence) + len(dangling)
    return {
        "total": total,
        "supported": supported,
        "no_evidence": no_evidence,
        "dangling": dangling,
        "unsupported_count": unsupported,
        "unsupported_rate": (unsupported / total) if total else 0.0,
    }


def keywords_for(kind: str, keyword_map: Optional[Mapping[str, Sequence[str]]] = None) -> List[str]:
    mapping = keyword_map if keyword_map is not None else DEFAULT_CONSTRAINT_KEYWORDS
    return list(mapping.get(kind, DEFAULT_CONSTRAINT_KEYWORDS.get(kind, [])))


def check_constraints(
    requirement: Mapping[str, Any],
    draft: Mapping[str, Any],
    *,
    keyword_map: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """逐条检查客户提出的约束，方案里有没有回应。

    只统计**可判定**的约束（有对应关键词的）；像 kind=other 这种没有关键词的，
    计入 total 但不计入分母 —— 不能把「判不了的」算成「没做到」。
    """
    text = draft_text(draft).lower()
    covered: List[Dict[str, Any]] = []
    missed: List[Dict[str, Any]] = []
    unjudgeable: List[Dict[str, Any]] = []

    for constraint in requirement.get("constraints", []) or []:
        kind = str(constraint.get("kind", "other"))
        keywords = keywords_for(kind, keyword_map)
        if not keywords:
            unjudgeable.append(constraint)
            continue
        hit = next((kw for kw in keywords if kw.lower() in text), None)
        record = {**constraint, "matched_keyword": hit}
        (covered if hit else missed).append(record)

    judgeable = len(covered) + len(missed)
    return {
        "total": judgeable + len(unjudgeable),
        "judgeable": judgeable,
        "covered": covered,
        "missed": missed,
        "unjudgeable": unjudgeable,
        "coverage_rate": (len(covered) / judgeable) if judgeable else 1.0,
    }


def review(
    draft: Mapping[str, Any],
    *,
    evidence_ids: Iterable[str] = (),
    requirement: Optional[Mapping[str, Any]] = None,
    keyword_map: Optional[Mapping[str, Sequence[str]]] = None,
    extra_issues: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """产出一份符合 Critique 契约的审校结论（规则部分）。

    **退回规则的核心口径：只退回「能改而没改」的问题。**

    - `dangling`（引用了不存在的证据）→ 退回。模型能改：要么找到真证据，要么老实标 EV-NONE。
      伪造引用比「承认没有证据」严重得多 —— 它伪装成了有依据。
    - 客户约束未回应 → 退回。这是漏项，能补。
    - `no_evidence`（显式标了 EV-NONE）→ **不退回**，只计入「无依据断言率」。
      它已经如实标注了「这条我没有证据」，改无可改；反复退回只会让流水线空转 ——
      真实模型下实测就是这个结果：连续 3 轮改写都在同样三条上打转。

    这个口径是有产品含义的：**售前的草稿本来就包含判断和假设**。
    把「判断」和「编造事实」混为一谈，等于逼模型要么撒谎、要么什么都不写。
    """
    claims = classify_claims(draft, evidence_ids)
    constraints = check_constraints(requirement or {}, draft, keyword_map=keyword_map)

    issues: List[Dict[str, Any]] = []
    for claim in claims["dangling"]:
        issues.append(
            {
                "type": "unsupported_claim",
                "detail": f"引用了不存在的证据 {claim.get('evidence_ref')}：{claim.get('text', '')[:40]}",
            }
        )
    for constraint in constraints["missed"]:
        issues.append(
            {
                "type": "uncovered_constraint",
                "detail": f"客户约束未被回应（{constraint.get('kind')}）：{constraint.get('detail')}",
            }
        )
    for issue in extra_issues or []:
        issues.append(dict(issue))

    return {
        "verdict": "pass" if not issues else "revise",
        "issues": issues,
        "checked_claims": claims["total"],
    }
