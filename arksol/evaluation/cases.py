"""用例加载与校验。

用例是**自建的、规模很小的**（20 条），这一点在 README、报告与面试稿里都必须写清楚。
它的作用是让三条流水线跑在**同一批输入**上，而不是充当行业基准。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from arksol.errors import ArksolError

REQUIRED_FIELDS = ("id", "input")


@dataclass
class Case:
    id: str
    input: str
    needs_followup: bool = False
    tags: Tuple[str, ...] = field(default_factory=tuple)
    note: str = ""

    @property
    def is_underspecified(self) -> bool:
        return self.needs_followup


def load_cases(path: Path | str) -> List[Case]:
    case_path = Path(path)
    if not case_path.exists():
        raise ArksolError(f"找不到用例文件：{case_path}")

    cases: List[Case] = []
    seen: set[str] = set()
    problems: List[str] = []

    for line_number, line in enumerate(case_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"第 {line_number} 行不是合法 JSON：{exc}")
            continue
        missing = [field_name for field_name in REQUIRED_FIELDS if not raw.get(field_name)]
        if missing:
            problems.append(f"第 {line_number} 行缺少字段 {missing}")
            continue
        case_id = str(raw["id"])
        if case_id in seen:
            problems.append(f"用例 id 重复：{case_id}")
            continue
        seen.add(case_id)
        cases.append(
            Case(
                id=case_id,
                input=str(raw["input"]),
                needs_followup=bool(raw.get("needs_followup", False)),
                tags=tuple(raw.get("tags") or ()),
                note=str(raw.get("note") or ""),
            )
        )

    if problems:
        raise ArksolError("用例文件校验未通过：\n  - " + "\n  - ".join(problems))
    if not cases:
        raise ArksolError(f"用例文件里没有任何用例：{case_path}")
    return cases


def split_by_followup(cases: List[Case]) -> Tuple[List[Case], List[Case]]:
    """拆成「信息充足」与「应当追问」两组，供追问命中率使用。"""
    enough = [case for case in cases if not case.needs_followup]
    need_ask = [case for case in cases if case.needs_followup]
    return enough, need_ask


def default_cases_path() -> Optional[Path]:
    """从当前工作目录往上找 evaluation/cases.jsonl，方便在任意子目录下运行。"""
    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / "evaluation" / "cases.jsonl"
        if candidate.exists():
            return candidate
    return None
