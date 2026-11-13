"""API FastAPI : /search (recherche) et /answer (reponse via Ollama, optionnel).

Interface web : http://127.0.0.1:8000/   (documentation de l'API : /docs)
Lancement : make index && make serve   (variable RAG_INDEX_DIR pour changer d'index)
/answer necessite un serveur Ollama local (OLLAMA_URL, defaut http://localhost:11434)
et un modele deja telecharge (OLLAMA_MODEL).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .retrieve import MODES

app = FastAPI(title="RAG hybride Docling", version="0.1.0")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
STATIC_DIR = Path(__file__).parent / "static"


@lru_cache(maxsize=1)
def get_retriever():
    from .index import Embedder, Index
    from .retrieve import Retriever

    index = Index.load(os.environ.get("RAG_INDEX_DIR", "index"))
    return Retriever(index, Embedder(index.embedder_name))


def _search(q: str, k: int, mode: str) -> list[dict]:
    try:
        retriever = get_retriever()
        hits = retriever.search(q, k=k, mode=mode)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    return [{"score": h.score, "doc": h.chunk.doc, "pages": h.chunk.pages,
             "section": h.chunk.section, "text": h.chunk.text} for h in hits]


@app.get("/", include_in_schema=False)
def home():
    """Interface web de recherche."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/search")
def search(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=50),
           mode: str = Query("hybrid")):
    if mode not in MODES:
        raise HTTPException(422, f"mode doit etre dans {MODES}")
    return {"query": q, "mode": mode, "results": _search(q, k, mode)}


class AnswerRequest(BaseModel):
    question: str
    k: int = 5
    mode: str = "hybrid"


@app.post("/answer")
def answer(req: AnswerRequest):
    if req.mode not in MODES:
        raise HTTPException(422, f"mode doit etre dans {MODES}")
    passages = _search(req.question, req.k, req.mode)
    context = "\n\n".join(f"[{i}] ({p['doc']}, pages {p['pages'] or 'n/a'})\n{p['text']}"
                          for i, p in enumerate(passages, start=1))
    prompt = ("Reponds en francais a la question en t'appuyant uniquement sur les passages ci-dessous. "
              "Cite les passages utilises par leur numero [n]. Si la reponse n'y figure pas, dis-le.\n\n"
              f"PASSAGES:\n{context}\n\nQUESTION: {req.question}\nREPONSE:")
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/generate", timeout=120,
                       json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False})
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Ollama indisponible ({OLLAMA_URL}, modele {OLLAMA_MODEL}): {e}") from e
    return {"answer": r.json().get("response", "").strip(), "sources": passages}
