"""Recherche : dense seul, BM25 seul, hybride par Reciprocal Rank Fusion (RRF),
avec reranking cross-encoder optionnel.

RRF (Cormack, Clarke & Buettcher, SIGIR 2009) : score(d) = somme sur les listes
de 1 / (k + rang_i(d)), rang_i commencant a 1. Ne depend que des rangs, donc
fusionne sans normalisation des scores dense (cosinus) et BM25.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .index import Index, tokenize

MODES = ("dense", "bm25", "hybrid", "hybrid+rerank")
DEFAULT_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fusionne des classements (listes d'identifiants, meilleur en premier).

    Retourne [(id, score)] trie par score decroissant ; a egalite, le meilleur rang
    obtenu dans une liste puis l'identifiant departagent (resultat deterministe).
    """
    if k < 0:
        raise ValueError("k doit etre >= 0")
    score: dict[int, float] = {}
    best: dict[int, int] = {}
    for ranking in rankings:
        seen = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:  # un doublon dans une meme liste ne compte qu'une fois
                continue
            seen.add(doc_id)
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (k + rank)
            best[doc_id] = min(best.get(doc_id, rank), rank)
    return sorted(score.items(), key=lambda kv: (-kv[1], best[kv[0]], kv[0]))


@dataclass
class Hit:
    idx: int
    score: float
    chunk: Chunk


class Retriever:
    def __init__(self, index: Index, embedder, rerank_model: str | None = None,
                 candidates: int = 50, rrf_k: int = 60):
        self.index, self.embedder = index, embedder
        self.candidates, self.rrf_k = candidates, rrf_k
        self.rerank_model = rerank_model or DEFAULT_RERANK_MODEL
        self._reranker = None

    # --- listes de base : (indices de chunks, scores), meilleur en premier ---
    def dense_ranking(self, query: str, n: int) -> tuple[list[int], list[float]]:
        q = self.embedder.encode_queries([query])
        scores, ids = self.index.faiss.search(q, min(n, len(self.index.chunks)))
        keep = ids[0] >= 0
        return ids[0][keep].tolist(), scores[0][keep].tolist()

    def bm25_ranking(self, query: str, n: int) -> tuple[list[int], list[float]]:
        s = np.asarray(self.index.bm25.get_scores(tokenize(query)))
        order = np.argsort(-s, kind="stable")[:n]
        order = [int(i) for i in order if s[i] > 0]  # score nul = aucun terme commun
        return order, [float(s[i]) for i in order]

    def _rerank(self, query: str, ids: list[int]) -> list[tuple[int, float]]:
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.rerank_model)
            except Exception as e:
                raise RuntimeError(f"Impossible de charger le reranker {self.rerank_model!r}: {e}") from e
        scores = self._reranker.predict([(query, self.index.chunks[i].text) for i in ids])
        return sorted(zip(ids, map(float, scores)), key=lambda t: -t[1])

    def search(self, query: str, k: int = 10, mode: str = "hybrid") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode inconnu: {mode!r} (attendu: {MODES})")
        n = max(self.candidates, k)
        if mode == "dense":
            ids, scores = self.dense_ranking(query, k)
            ranked = list(zip(ids, scores))
        elif mode == "bm25":
            ids, scores = self.bm25_ranking(query, k)
            ranked = list(zip(ids, scores))
        else:
            fused = rrf_fuse([self.dense_ranking(query, n)[0], self.bm25_ranking(query, n)[0]], self.rrf_k)
            if mode == "hybrid":
                ranked = fused[:k]
            else:
                ranked = self._rerank(query, [i for i, _ in fused[:n]])[:k]
        return [Hit(i, s, self.index.chunks[i]) for i, s in ranked[:k]]
