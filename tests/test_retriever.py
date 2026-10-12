"""检索测试。中文部分靠字符二元组，所以用例要覆盖「问句 vs 条目措辞不一致」的情况。"""

from __future__ import annotations

import unittest

from arksol.kb import load_default_kb
from arksol.tools.retriever import BM25Index, best_terms, build_index, tokenize


class TestTokenize(unittest.TestCase):
    def test_chinese_bigrams(self) -> None:
        self.assertEqual(tokenize("知识库"), ["知识", "识库"])

    def test_single_chinese_char(self) -> None:
        self.assertEqual(tokenize("好"), ["好"])

    def test_ascii_words_lowered(self) -> None:
        self.assertEqual(tokenize("OpenAI SDK"), ["openai", "sdk"])

    def test_version_like_token_kept_whole(self) -> None:
        self.assertIn("v4.1-flash", tokenize("deepseek v4.1-flash"))

    def test_mixed_text(self) -> None:
        tokens = tokenize("方舟 API 404")
        self.assertIn("404", tokens)
        self.assertIn("方舟", tokens)
        self.assertIn("api", tokens)

    def test_empty_text(self) -> None:
        self.assertEqual(tokenize(""), [])


class TestBM25(unittest.TestCase):
    def setUp(self) -> None:
        self.index = BM25Index(
            [
                ("d1", "企业知识库问答：数据不能出内网时要考虑私有化部署"),
                ("d2", "文档批量审核：要测重复运行的一致性，不能只测准确率"),
                ("d3", "代码助手：把整个仓库塞进上下文成本很高"),
            ]
        )

    def test_finds_relevant_doc(self) -> None:
        hits = self.index.search("数据不出内网")
        self.assertTrue(hits)
        self.assertEqual(hits[0].doc_id, "d1")

    def test_ranks_by_score_desc(self) -> None:
        hits = self.index.search("上下文 成本")
        scores = [hit.score for hit in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_respects_top_k(self) -> None:
        self.assertLessEqual(len(self.index.search("成本", top_k=2)), 2)

    def test_no_match_returns_empty(self) -> None:
        self.assertEqual(self.index.search("量子纠缠"), [])

    def test_empty_query_returns_empty(self) -> None:
        self.assertEqual(self.index.search(""), [])

    def test_deterministic_order(self) -> None:
        first = [hit.doc_id for hit in self.index.search("成本", top_k=3)]
        second = [hit.doc_id for hit in self.index.search("成本", top_k=3)]
        self.assertEqual(first, second)

    def test_matched_terms_recorded(self) -> None:
        hits = self.index.search("知识库")
        self.assertTrue(hits[0].matched_terms)

    def test_empty_corpus(self) -> None:
        empty = BM25Index([])
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty.search("任意"), [])


class TestRetrievalOverRealKB(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = load_default_kb()
        cls.index = build_index(cls.kb)

    def test_index_size_matches_kb(self) -> None:
        self.assertEqual(len(self.index), len(self.kb.entries))

    def test_chinese_scenario_query_hits_pattern_entry(self) -> None:
        hits = self.index.search("知识库问答")
        self.assertTrue(hits)
        self.assertEqual(hits[0].doc_id, "pattern.enterprise-kb-qa")

    def test_error_code_query_surfaces_error_entries(self) -> None:
        # 「404」这个词既出现在错误码条目里，也出现在「最常见的 404 来源」那条指南里，
        # 所以这里断言的是「错误码条目**出现在结果中**」，而不是「结果全是错误码条目」
        hits = self.index.search("404 InvalidEndpoint", top_k=5)
        self.assertTrue(hits)
        self.assertTrue(any(hit.doc_id.startswith("err.") for hit in hits))

    def test_endpoint_confusion_query_hits_guide(self) -> None:
        hits = self.index.search("推理接入点 还是 Model ID", top_k=3)
        self.assertIn("guide.endpoint-vs-model-id", [hit.doc_id for hit in hits])

    def test_base_url_query_hits_guide(self) -> None:
        hits = self.index.search("鉴权 404", top_k=3)
        self.assertIn("guide.base-url-and-auth", [hit.doc_id for hit in hits])

    def test_free_quota_query_hits_guide(self) -> None:
        hits = self.index.search("免费额度", top_k=3)
        self.assertIn("guide.free-quota", [hit.doc_id for hit in hits])

    def test_hits_resolve_back_to_kb_entries(self) -> None:
        for hit in self.index.search("成本 缓存", top_k=3):
            entry = self.kb.get(hit.doc_id)
            self.assertIn("sources", entry)

    def test_best_terms_helper(self) -> None:
        hits = self.index.search("计费 缓存")
        self.assertTrue(best_terms(hits))


if __name__ == "__main__":
    unittest.main()
