"""Index + recherche de bout en bout avec l'embedder de test (sans telechargement)."""
import pytest

from src.chunking import Chunk
from src.index import HashingEmbedder, Index, tokenize
from src.retrieve import Retriever

DOCS = [
    "Les penalites de retard sont calculees par jour calendaire",
    "Le prix du marche est ferme et actualisable",
    "La garantie de parfait achevement dure un an",
    "Les delais de paiement sont de trente jours",
    "Le titulaire fournit une attestation d'assurance",
]


@pytest.fixture
def retriever():
    chunks = [Chunk(f"d::{i}", "d", t, pages=[i + 1]) for i, t in enumerate(DOCS)]
    emb = HashingEmbedder()
    return Retriever(Index.build(chunks, emb), emb)


def test_tokenize_lowercase_accents_stopwords():
    assert tokenize("Le Délai de l'Assurance") == ["délai", "assurance"]


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_exact_term_query_finds_chunk_first(retriever, mode):
    hits = retriever.search("garantie de parfait achevement", k=3, mode=mode)
    assert hits[0].chunk.pages == [3]


def test_bm25_returns_nothing_without_common_term(retriever):
    assert retriever.search("xyzzy plugh", k=3, mode="bm25") == []


def test_hybrid_returns_at_most_k_unique(retriever):
    hits = retriever.search("prix et delais", k=3, mode="hybrid")
    ids = [h.idx for h in hits]
    assert len(ids) <= 3 and len(set(ids)) == len(ids)


def test_unknown_mode(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="magic")


def test_save_load_roundtrip(retriever, tmp_path):
    retriever.index.save(tmp_path)
    loaded = Index.load(tmp_path)
    r2 = Retriever(loaded, HashingEmbedder())
    assert [c.text for c in loaded.chunks] == DOCS
    assert r2.search("delais de paiement", k=1, mode="hybrid")[0].chunk.pages == [4]


def test_empty_index_rejected():
    import numpy as np
    with pytest.raises(ValueError):
        Index([], np.zeros((0, 4), dtype="float32"), "x")
