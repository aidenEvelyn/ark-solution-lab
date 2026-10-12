"""知识库：人工维护、**每条都标来源**、加载时做完整性校验。

这个模块存在的理由，是为了让「不许编」变成一条能被机器检查的规则：

- 每条知识都必须挂至少一个已登记的来源；
- 来源必须写明 URL、读取日期、可信度；
- 引用一个没登记过的来源 id → 加载直接失败，而不是悄悄放过。

知识库里的事实**不是官方文档的替代品**，接真实业务前必须重新核对 ——
这条写在每个来源的 `note` 里，也写在 README 的局限里。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from arksol.errors import KBError

KB_DIR = Path(__file__).resolve().parent

_ENTRY_FILES = {
    "model": "models.json",  # 模型选型
    "error": "errors.json",  # 报错与排查
    "pattern": "patterns.json",  # 方案模式
}

_REQUIRED_SOURCE_FIELDS = ("title", "url", "read_at", "confidence")
_VALID_CONFIDENCE = ("高", "中", "低")


@dataclass
class KnowledgeBase:
    sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    entries: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- 加载 ---------------------------------------------------------------------------

    @classmethod
    def load(cls, kb_dir: Optional[Path] = None) -> "KnowledgeBase":
        base = Path(kb_dir) if kb_dir is not None else KB_DIR

        sources_path = base / "sources.json"
        if not sources_path.exists():
            raise KBError(f"找不到来源登记表：{sources_path}")
        sources_doc = json.loads(sources_path.read_text(encoding="utf-8"))
        sources = sources_doc.get("sources", {})

        entries: List[Dict[str, Any]] = []
        for kind, filename in _ENTRY_FILES.items():
            path = base / filename
            if not path.exists():
                raise KBError(f"找不到知识库文件：{path}")
            doc = json.loads(path.read_text(encoding="utf-8"))
            for entry in doc.get("entries", []):
                merged = dict(entry)
                merged.setdefault("kind", kind)
                merged.setdefault("source_file", filename)
                entries.append(merged)

        kb = cls(sources=sources, entries=entries, meta={"dir": str(base), **sources_doc.get("meta", {})})
        kb.validate_integrity()
        return kb

    # -- 完整性校验 ---------------------------------------------------------------------

    def validate_integrity(self) -> None:
        problems: List[str] = []

        if not self.sources:
            problems.append("来源登记表是空的")
        for source_id, source in self.sources.items():
            for key in _REQUIRED_SOURCE_FIELDS:
                if not source.get(key):
                    problems.append(f"来源 {source_id} 缺少字段 {key}")
            confidence = source.get("confidence")
            if confidence and confidence not in _VALID_CONFIDENCE:
                problems.append(f"来源 {source_id} 的可信度 {confidence!r} 不在 {_VALID_CONFIDENCE}")

        seen_ids: set[str] = set()
        for entry in self.entries:
            entry_id = entry.get("id")
            if not entry_id:
                problems.append(f"知识条目缺少 id（来自 {entry.get('source_file')}）")
                continue
            if entry_id in seen_ids:
                problems.append(f"知识条目 id 重复：{entry_id}")
            seen_ids.add(entry_id)

            if not str(entry.get("text", "")).strip():
                problems.append(f"知识条目 {entry_id} 的 text 是空的")

            refs = entry.get("sources") or []
            if not refs:
                problems.append(f"知识条目 {entry_id} 没有挂任何来源（这是硬规则）")
            for ref in refs:
                if ref not in self.sources:
                    problems.append(f"知识条目 {entry_id} 引用了未登记的来源 {ref}")

        if problems:
            raise KBError("知识库完整性校验未通过：\n  - " + "\n  - ".join(problems))

    # -- 查询 ---------------------------------------------------------------------------

    def get(self, entry_id: str) -> Dict[str, Any]:
        for entry in self.entries:
            if entry.get("id") == entry_id:
                return entry
        raise KeyError(f"知识库里没有条目 {entry_id!r}")

    def by_kind(self, kind: str) -> List[Dict[str, Any]]:
        return [e for e in self.entries if e.get("kind") == kind]

    def source_of(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """把条目的来源 id 展开成来源详情，方便写进证据里。"""
        return [{"id": sid, **self.sources[sid]} for sid in entry.get("sources", []) if sid in self.sources]

    def documents(self) -> List[Tuple[str, str]]:
        """给检索器用的 (doc_id, text) 列表 —— 标题、标签、正文都参与检索。"""
        docs: List[Tuple[str, str]] = []
        for entry in self.entries:
            parts = [
                str(entry.get("title", "")),
                " ".join(entry.get("tags", [])),
                str(entry.get("text", "")),
            ]
            docs.append((entry["id"], "\n".join(p for p in parts if p)))
        return docs

    def stats(self) -> Dict[str, Any]:
        by_kind: Dict[str, int] = {}
        for entry in self.entries:
            by_kind[entry.get("kind", "?")] = by_kind.get(entry.get("kind", "?"), 0) + 1
        confidences: Dict[str, int] = {}
        for source in self.sources.values():
            key = str(source.get("confidence", "?"))
            confidences[key] = confidences.get(key, 0) + 1
        return {"sources": len(self.sources), "entries": len(self.entries), "by_kind": by_kind, "source_confidence": confidences}


def load_default_kb() -> KnowledgeBase:
    """加载随包分发的知识库。"""
    return KnowledgeBase.load()
