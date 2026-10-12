"""全链路 trace：每个环节的输入、输出、耗时、token、校验结果都写成 JSONL。

出问题时能定位到**具体哪一步**，而不是「它有时候就是不行」。
JSONL 而不是一个大 JSON：可以边跑边追加，跑崩了也不丢已记录的部分。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


class Tracer:
    """JSONL 追加写入器。用 with 语句或显式 close()。"""

    def __init__(self, path: Optional[os.PathLike | str] = None, *, run_id: Optional[str] = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.path: Optional[Path] = Path(path) if path is not None else None
        self._fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8", newline="\n")

    # -- 写入 ---------------------------------------------------------------------------

    def emit(
        self,
        *,
        agent: str,
        event: str,
        payload: Optional[Any] = None,
        usage: Optional[dict] = None,
        elapsed_ms: Optional[float] = None,
        level: str = "info",
    ) -> dict:
        record = {
            "ts": _now_iso(),
            "run_id": self.run_id,
            "agent": agent,
            "event": event,
            "level": level,
        }
        if payload is not None:
            record["payload"] = payload
        if usage is not None:
            record["usage"] = usage
        if elapsed_ms is not None:
            record["elapsed_ms"] = round(elapsed_ms, 1)

        if self._fh is not None:
            self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        return record

    def span(self, *, agent: str, event: str = "call", payload: Optional[Any] = None):
        """计时上下文：with tracer.span(agent="intake"): ..."""
        return _Span(self, agent=agent, event=event, payload=payload)

    # -- 生命周期 -----------------------------------------------------------------------

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class _Span:
    def __init__(self, tracer: Tracer, *, agent: str, event: str, payload: Optional[Any]) -> None:
        self.tracer = tracer
        self.agent = agent
        self.event = event
        self.payload = payload
        self.started = 0.0

    def __enter__(self) -> "_Span":
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed_ms = (time.perf_counter() - self.started) * 1000
        self.tracer.emit(
            agent=self.agent,
            event=self.event,
            payload=self.payload,
            elapsed_ms=elapsed_ms,
            level="error" if exc_type is not None else "info",
        )


def read_trace(path: os.PathLike | str) -> List[dict]:
    """读回 JSONL；空行与坏行跳过（跑崩时最后一行可能是半截的）。"""
    records: List[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def summarize(records: Iterable[dict]) -> dict:
    """给评测报告用的汇总：环节数、错误数、总耗时、按 agent 分组。"""
    records = list(records)
    by_agent: dict[str, int] = {}
    total_ms = 0.0
    errors = 0
    for record in records:
        by_agent[record.get("agent", "?")] = by_agent.get(record.get("agent", "?"), 0) + 1
        total_ms += float(record.get("elapsed_ms") or 0.0)
        if record.get("level") == "error":
            errors += 1
    return {
        "records": len(records),
        "errors": errors,
        "total_elapsed_ms": round(total_ms, 1),
        "by_agent": by_agent,
    }
