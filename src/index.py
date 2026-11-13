"""Index dense (FAISS) + index lexical (BM25) construits sur les memes chunks.

Usage : python -m src.index --processed data/processed --strategy section --size 256 --out index
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from .chunking import STRATEGIES, Chunk, chunk_corpus
from .extract import load_corpus

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

_STOPWORDS = frozenset(
    "le la les l un une des du de d et ou en au aux a à est sont ce cette ces se sa son ses "
    "que qui quoi dans par pour sur avec sans sous il elle ils elles on ne pas plus ou où "
    "y s c n qu j m t".split())
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenisation BM25 : NFKC, minuscules, mots alphanumeriques, sans mots vides."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return [t for t in _TOKEN_RE.findall(text) if t not in _STOPWORDS]


class EmbedderLoadError(RuntimeError):
    pass


class Embedder:
    """Enveloppe sentence-transformers (prefixes query:/passage: pour la famille e5)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        except Exception as e:
            raise EmbedderLoadError(
                f"Impossible de charger le modele d'embeddings {model_name!r} "
                f"({type(e).__name__}: {e}). Verifiez l'acces a huggingface.co "
                f"(ou un cache local HF_HOME) puis relancez.") from e
        self._e5 = "e5" in model_name.lower()

    def _encode(self, texts: list[str]) -> np.ndarray:
        v = self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        return np.ascontiguousarray(v, dtype="float32")

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"passage: {t}" for t in texts] if self._e5 else texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"query: {t}" for t in texts] if self._e5 else texts)


class HashingEmbedder:
    """Embedder deterministe par hachage de mots. UNIQUEMENT pour les tests unitaires
    (pas de telechargement de modele) : n'est jamais utilise par evaluate.py."""

    model_name = "hashing-test-only"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]).astype("float32")

    encode_queries = encode_passages


class Index:
    """Chunks + index FAISS (produit scalaire sur vecteurs normalises = cosinus) + BM25."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder_name: str):
        import faiss
        from rank_bm25 import BM25Okapi

        if len(chunks) == 0:
            raise ValueError("Aucun chunk a indexer.")
        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings et chunks de longueurs differentes")
        self.chunks = chunks
        self.embedder_name = embedder_name
        self.faiss = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss.add(np.ascontiguousarray(embeddings, dtype="float32"))
        self.embeddings = embeddings
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    @classmethod
    def build(cls, chunks: list[Chunk], embedder) -> "Index":
        return cls(chunks, embedder.encode_passages([c.text for c in chunks]), embedder.model_name)

    # --- persistance : chunks.jsonl + embeddings.npy (BM25 et FAISS sont reconstruits) ---
    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(out / "embeddings.npy", self.embeddings)
        (out / "meta.json").write_text(json.dumps({"embedder": self.embedder_name}), encoding="utf-8")

    @classmethod
    def load(cls, in_dir: str | Path) -> "Index":
        d = Path(in_dir)
        if not (d / "chunks.jsonl").exists():
            raise FileNotFoundError(f"Index introuvable dans {d} (lancez `make index`).")
        chunks = [Chunk(**json.loads(l)) for l in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l]
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return cls(chunks, np.load(d / "embeddings.npy"), meta["embedder"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--strategy", choices=STRATEGIES, default="section")
    ap.add_argument("--size", type=int, default=256, help="taille max d'un chunk, en mots")
    ap.add_argument("--overlap", type=int, default=50, help="recouvrement, en mots")
    ap.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    ap.add_argument("--out", default="index")
    a = ap.parse_args()

    corpus = load_corpus(a.processed)
    if not corpus:
        raise SystemExit(f"Aucun document extrait dans {a.processed} (lancez `make extract`).")
    chunks = chunk_corpus(corpus, a.strategy, a.size, a.overlap)
    idx = Index.build(chunks, Embedder(a.model))
    idx.save(a.out)
    print(f"{len(chunks)} chunks indexes ({a.strategy}, size={a.size}, overlap={a.overlap}) -> {a.out}/")
