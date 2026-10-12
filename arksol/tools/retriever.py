"""纯 Python 的 BM25 检索。

为什么不用向量检索：这个项目的知识库只有二三十条，人工维护、术语密集
（「上下文长度」「RPM」「数据不出内网」），关键词检索在这个规模上完全够用；
引入 embedding 会同时引入 API 依赖、额外成本和一个新的不确定性来源。
**在规模没到之前，不引入向量库是一种自觉的取舍**，不是技术落后。

中文没有分词器怎么办：不做词切分，用**字符二元组（bigram）**。
「上下文长度」→ 上、下文、文长、长度、度。这是无分词器中文检索的常用做法，
代价是索引变大、语义精度略低 —— 在这个规模上可以接受。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

_ASCII_RE = re.compile(r"[a-z0-9]+(?:[._+-][a-z0-9]+)*")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    """把文本切成检索用的词元：ASCII 按词，CJK 按字符二元组。"""
    if not text:
        return []
    lowered = text.lower()
    tokens: List[str] = list(_ASCII_RE.findall(lowered))
    for run in _CJK_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass
class SearchHit:
    doc_id: str
    score: float
    matched_terms: List[str] = field(default_factory=list)


class BM25Index:
    """标准 BM25（k1=1.5, b=0.75）。文档量级：几十到几千条。"""

    def __init__(self, docs: Iterable[Tuple[str, str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self._tf: Dict[str, Counter] = {}
        self._len: Dict[str, int] = {}
        self._df: Counter = Counter()

        for doc_id, text in docs:
            tokens = tokenize(text)
            self.doc_ids.append(doc_id)
            counts = Counter(tokens)
            self._tf[doc_id] = counts
            self._len[doc_id] = max(1, len(tokens))
            for term in counts:
                self._df[term] += 1

        self._n = len(self.doc_ids)
        self._avgdl = (sum(self._len.values()) / self._n) if self._n else 0.0

    def __len__(self) -> int:
        return self._n

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        # 带 +1 的写法保证 idf 非负，避免高频词把分数拉负
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        if self._n == 0:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: Dict[str, float] = {}
        matched: Dict[str, List[str]] = {}

        for term in set(query_terms):
            idf = self._idf(term)
            if idf <= 0.0:
                continue
            for doc_id in self.doc_ids:
                tf = self._tf[doc_id].get(term, 0)
                if tf == 0:
                    continue
                norm = 1.0 - self.b + self.b * (self._len[doc_id] / self._avgdl if self._avgdl else 1.0)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
                matched.setdefault(doc_id, []).append(term)

        hits = [SearchHit(doc_id=d, score=round(s, 6), matched_terms=sorted(set(matched[d]))) for d, s in scores.items()]
        # 分数相同时按 doc_id 排序，保证输出可复现（评测要求结果稳定）
        hits.sort(key=lambda h: (-h.score, h.doc_id))
        return hits[: max(0, top_k)]


def build_index(kb) -> BM25Index:
    """从知识库建索引。接受任何提供 documents() 的对象。"""
    return BM25Index(kb.documents())


def best_terms(hits: Sequence[SearchHit], limit: int = 10) -> List[str]:
    """汇总命中的词元，供 trace 与调试使用。"""
    seen: List[str] = []
    for hit in hits:
        for term in hit.matched_terms:
            if term not in seen:
                seen.append(term)
            if len(seen) >= limit:
                return seen
    return seen
