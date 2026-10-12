"""知识库测试。重点是**完整性规则真的会拦住不合格的知识**。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from arksol.errors import KBError
from arksol.kb import KnowledgeBase, load_default_kb

VALID_SOURCE = {
    "S01": {"title": "示例来源", "url": "https://example.com", "read_at": "2026-10-12", "confidence": "中"}
}


def write_kb(base: Path, sources: dict, model_entries: list) -> None:
    (base / "sources.json").write_text(
        json.dumps({"meta": {"updated": "2026-10-12"}, "sources": sources}, ensure_ascii=False), encoding="utf-8"
    )
    (base / "models.json").write_text(json.dumps({"entries": model_entries}, ensure_ascii=False), encoding="utf-8")
    (base / "errors.json").write_text(json.dumps({"entries": []}), encoding="utf-8")
    (base / "patterns.json").write_text(json.dumps({"entries": []}), encoding="utf-8")


class TestDefaultKB(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = load_default_kb()

    def test_loads_without_error(self) -> None:
        self.assertGreater(len(self.kb.entries), 0)
        self.assertGreater(len(self.kb.sources), 0)

    def test_every_entry_cites_registered_source(self) -> None:
        for entry in self.kb.entries:
            refs = entry.get("sources") or []
            self.assertTrue(refs, f"{entry['id']} 没挂来源")
            for ref in refs:
                self.assertIn(ref, self.kb.sources, f"{entry['id']} 引用了未登记来源 {ref}")

    def test_stats_shape(self) -> None:
        stats = self.kb.stats()
        self.assertEqual(stats["entries"], len(self.kb.entries))
        for kind in ("model", "error", "pattern"):
            self.assertIn(kind, stats["by_kind"])

    def test_get_unknown_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.kb.get("does.not.exist")

    def test_get_known_entry(self) -> None:
        entry = self.kb.get("guide.base-url-and-auth")
        self.assertIn("ark.cn-beijing.volces.com", entry["text"])

    def test_documents_are_non_empty(self) -> None:
        docs = self.kb.documents()
        self.assertEqual(len(docs), len(self.kb.entries))
        for doc_id, text in docs:
            self.assertTrue(doc_id)
            self.assertTrue(text.strip())

    def test_source_of_expands_ids(self) -> None:
        entry = self.kb.get("guide.free-quota")
        expanded = self.kb.source_of(entry)
        self.assertTrue(expanded)
        self.assertIn("url", expanded[0])

    def test_disclaimer_present(self) -> None:
        # 知识库必须自带免责声明，否则它会被当成官方文档用
        self.assertIn("disclaimer", self.kb.meta)


class TestIntegrityRules(unittest.TestCase):
    def test_rejects_entry_without_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_kb(
                base,
                VALID_SOURCE,
                [{"id": "model.x", "title": "x", "text": "正文", "sources": []}],
            )
            with self.assertRaises(KBError) as ctx:
                KnowledgeBase.load(base)
            self.assertIn("没有挂任何来源", str(ctx.exception))

    def test_rejects_dangling_source_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_kb(
                base,
                VALID_SOURCE,
                [{"id": "model.x", "title": "x", "text": "正文", "sources": ["S99"]}],
            )
            with self.assertRaises(KBError) as ctx:
                KnowledgeBase.load(base)
            self.assertIn("S99", str(ctx.exception))

    def test_rejects_duplicate_entry_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            entry = {"id": "model.x", "title": "x", "text": "正文", "sources": ["S01"]}
            write_kb(base, VALID_SOURCE, [entry, dict(entry)])
            with self.assertRaises(KBError) as ctx:
                KnowledgeBase.load(base)
            self.assertIn("id 重复", str(ctx.exception))

    def test_rejects_empty_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_kb(base, VALID_SOURCE, [{"id": "model.x", "title": "x", "text": "  ", "sources": ["S01"]}])
            with self.assertRaises(KBError) as ctx:
                KnowledgeBase.load(base)
            self.assertIn("text 是空的", str(ctx.exception))

    def test_rejects_source_missing_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            bad = {"S01": {"title": "无 URL", "read_at": "2026-10-12", "confidence": "中"}}
            write_kb(base, bad, [{"id": "model.x", "title": "x", "text": "正文", "sources": ["S01"]}])
            with self.assertRaises(KBError) as ctx:
                KnowledgeBase.load(base)
            self.assertIn("缺少字段 url", str(ctx.exception))

    def test_rejects_unknown_confidence_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            bad = {
                "S01": {"title": "x", "url": "https://example.com", "read_at": "2026-10-12", "confidence": "很高"}
            }
            write_kb(base, bad, [{"id": "model.x", "title": "x", "text": "正文", "sources": ["S01"]}])
            with self.assertRaises(KBError) as ctx:
                KnowledgeBase.load(base)
            self.assertIn("可信度", str(ctx.exception))

    def test_accepts_valid_minimal_kb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_kb(base, VALID_SOURCE, [{"id": "model.x", "title": "标题", "text": "正文", "sources": ["S01"]}])
            kb = KnowledgeBase.load(base)
            self.assertEqual(len(kb.entries), 1)
            self.assertEqual(kb.entries[0]["kind"], "model")  # kind 按所在文件自动补上


if __name__ == "__main__":
    unittest.main()
