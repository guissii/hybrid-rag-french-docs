import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import src.app as app_module
from src.chunking import Chunk
from src.index import HashingEmbedder, Index
from src.retrieve import Retriever


@pytest.fixture
def client(monkeypatch):
    chunks = [Chunk("d::0", "d.pdf", "Les delais de paiement sont de trente jours", pages=[4], section="Paiement"),
              Chunk("d::1", "d.pdf", "La garantie dure un an", pages=[7], section="Garantie")]
    emb = HashingEmbedder()
    r = Retriever(Index.build(chunks, emb), emb)
    monkeypatch.setattr(app_module, "get_retriever", lambda: r)
    return TestClient(app_module.app)


def test_home_serves_interface(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Recherche documentaire" in r.text


def test_search_returns_passages(client):
    r = client.get("/search", params={"q": "delais de paiement", "k": 1, "mode": "hybrid"})
    assert r.status_code == 200
    top = r.json()["results"][0]
    assert top["doc"] == "d.pdf" and top["pages"] == [4] and top["section"] == "Paiement"


def test_search_rejects_bad_mode(client):
    assert client.get("/search", params={"q": "x", "mode": "magic"}).status_code == 422


def test_missing_index_gives_503(monkeypatch):
    def boom():
        raise FileNotFoundError("Index introuvable")
    monkeypatch.setattr(app_module, "get_retriever", boom)
    assert TestClient(app_module.app).get("/search", params={"q": "x"}).status_code == 503


def test_answer_reports_ollama_down(client, monkeypatch):
    monkeypatch.setattr(app_module, "OLLAMA_URL", "http://127.0.0.1:9")  # port ferme
    r = client.post("/answer", json={"question": "delais de paiement", "k": 1})
    assert r.status_code == 502 and "Ollama" in r.json()["detail"]
