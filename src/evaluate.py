"""Evaluation : balayage (strategie de chunking x taille x mode de recherche).

Metriques : hit@k, MRR, nDCG@10 (pertinence binaire), latence en ms/requete
(encodage de la requete inclus, indexation exclue).
Sorties : results/results.csv, results/results.md, et le tableau injecte dans le
README entre les marqueurs RESULTS:START / RESULTS:END.

AUCUN chiffre n'est produit tant que data/eval/questions.jsonl n'est pas
completement etiquete (voir data/eval/README.md pour le format).

Usage : python -m src.evaluate --sizes 128 256 --strategies fixed section clause [--rerank]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .chunking import STRATEGIES, Chunk, chunk_corpus
from .extract import Element, load_corpus

MIN_QUESTIONS = 20
KS = (1, 3, 5, 10)


class LabelError(Exception):
    """Jeu d'evaluation absent, vide, non etiquete ou incoherent avec le corpus."""


# ----------------------------------------------------------------------------
# Metriques (pertinence binaire ; `rels[i]` = le resultat de rang i+1 est pertinent)
# ----------------------------------------------------------------------------
def hit_at_k(rels: list[bool], k: int) -> float:
    return 1.0 if any(rels[:k]) else 0.0


def reciprocal_rank(rels: list[bool]) -> float:
    for i, r in enumerate(rels, start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: list[bool], k: int, n_relevant: int) -> float:
    """nDCG@k binaire. `n_relevant` = nombre total de chunks pertinents dans le corpus
    (borne l'DCG ideal). Retourne 0 si aucun chunk n'est pertinent."""
    if n_relevant <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(rels[:k], start=1) if r)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_relevant, k) + 1))
    return dcg / idcg


# ----------------------------------------------------------------------------
# Questions etiquetees
# ----------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip().casefold()


@dataclass
class Question:
    id: str
    question: str
    relevant: list[dict]  # {"doc": ..., "pages": [...]} et/ou {"doc": ..., "evidence": "..."}


def _is_unlabeled(q: dict) -> bool:
    text = (q.get("question") or "").strip()
    return (not text or text.upper().startswith("TODO") or not q.get("relevant")
            or any(not (r.get("pages") or (r.get("evidence") or "").strip()) or not r.get("doc")
                   or str(r.get("evidence", "")).upper().startswith("TODO")
                   for r in q["relevant"]))


def load_questions(path: str | Path, corpus: dict[str, list[Element]]) -> list[Question]:
    path = Path(path)
    if not path.exists():
        raise LabelError(f"{path} introuvable.")
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise LabelError(f"{path}:{n}: JSON invalide ({e})") from e
    if not rows:
        raise LabelError(f"{path} est vide : etiquetez vos questions avant d'evaluer.")
    unlabeled = [r.get("id", f"ligne {i}") for i, r in enumerate(rows, 1) if _is_unlabeled(r)]
    if unlabeled:
        raise LabelError(
            f"{len(unlabeled)}/{len(rows)} questions non etiquetees (question vide/TODO ou 'relevant' "
            f"incomplet) : {', '.join(map(str, unlabeled[:10]))}{' ...' if len(unlabeled) > 10 else ''}. "
            f"Completez data/eval/questions.jsonl (format: data/eval/README.md) ; "
            f"aucune metrique n'est calculee sur un jeu non etiquete.")
    if len(rows) < MIN_QUESTIONS:
        raise LabelError(f"{len(rows)} questions : il en faut au moins {MIN_QUESTIONS}.")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise LabelError("Identifiants de questions dupliques.")

    qs = []
    for r in rows:
        for rel in r["relevant"]:
            doc = rel["doc"]
            if doc not in corpus:
                raise LabelError(f"{r['id']}: document {doc!r} absent du corpus extrait "
                                 f"({', '.join(sorted(corpus)) or 'vide'}).")
            els = corpus[doc]
            pages = {e.page for e in els if e.page is not None}
            for p in rel.get("pages") or []:
                if p not in pages:
                    raise LabelError(f"{r['id']}: page {p} inexistante dans {doc}.")
            ev = (rel.get("evidence") or "").strip()
            if ev and not any(_norm(ev) in _norm(e.text) for e in els):
                raise LabelError(f"{r['id']}: l'extrait 'evidence' est introuvable dans {doc} "
                                 f"(il doit etre copie tel quel, dans UN seul paragraphe/tableau).")
        qs.append(Question(r["id"], r["question"].strip(), r["relevant"]))
    return qs


def relevant_chunk_ids(q: Question, chunks: list[Chunk], corpus: dict[str, list[Element]]) -> set[int]:
    """Indices des chunks pertinents : meme document et (page en commun OU element 'evidence' couvert)."""
    out: set[int] = set()
    for rel in q.relevant:
        doc = rel["doc"]
        pages = set(rel.get("pages") or [])
        ev = _norm(rel.get("evidence") or "")
        ev_elems = {e.idx for e in corpus[doc] if ev and ev in _norm(e.text)}
        for i, c in enumerate(chunks):
            if c.doc != doc:
                continue
            if (pages and pages & set(c.pages)) or (ev_elems and ev_elems & set(c.elem_ids)):
                out.add(i)
    return out


# ----------------------------------------------------------------------------
# Balayage
# ----------------------------------------------------------------------------
def evaluate_config(retriever, questions, relevant_sets, mode: str) -> dict:
    retriever.search(questions[0].question, k=10, mode=mode)  # chauffe (chargement paresseux, caches)
    per_q, lat = [], []
    for q, rel_ids in zip(questions, relevant_sets):
        t0 = time.perf_counter()
        hits = retriever.search(q.question, k=10, mode=mode)
        lat.append((time.perf_counter() - t0) * 1000.0)
        rels = [h.idx in rel_ids for h in hits]
        rels += [False] * (10 - len(rels))  # moins de 10 resultats = rangs vides non pertinents
        per_q.append((rels, len(rel_ids)))
    row = {f"hit@{k}": statistics.fmean(hit_at_k(r, k) for r, _ in per_q) for k in KS}
    row["mrr"] = statistics.fmean(reciprocal_rank(r) for r, _ in per_q)
    row["ndcg@10"] = statistics.fmean(ndcg_at_k(r, 10, n) for r, n in per_q)
    row["latency_ms_mean"] = statistics.fmean(lat)
    row["latency_ms_p95"] = sorted(lat)[max(0, math.ceil(0.95 * len(lat)) - 1)]
    return row


COLUMNS = ["strategy", "size", "overlap", "mode", "n_chunks", "n_questions",
           *[f"hit@{k}" for k in KS], "mrr", "ndcg@10", "latency_ms_mean", "latency_ms_p95"]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: (round(r[c], 4) if isinstance(r[c], float) else r[c]) for c in COLUMNS})


def markdown_table(rows: list[dict]) -> str:
    head = ["Chunking", "Taille", "Recouvr.", "Mode", "Chunks", "hit@1", "hit@3", "hit@5", "hit@10",
            "MRR", "nDCG@10", "Latence moy. (ms)", "Latence p95 (ms)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join([
            r["strategy"], str(r["size"]), str(r["overlap"]), r["mode"], str(r["n_chunks"]),
            *[f"{r[f'hit@{k}']:.3f}" for k in KS], f"{r['mrr']:.3f}", f"{r['ndcg@10']:.3f}",
            f"{r['latency_ms_mean']:.1f}", f"{r['latency_ms_p95']:.1f}"]) + " |")
    return "\n".join(lines)


def write_markdown(rows, path: Path, meta: dict) -> str:
    table = markdown_table(rows)
    md = f"""# Résultats d'évaluation

Fichier généré automatiquement par `src/evaluate.py` — ne pas éditer à la main.

- Date de génération : {meta['date']}
- Commande : `{meta['command']}`
- Modèle d'embeddings : `{meta['embedder']}`
- Reranker : {('`' + meta['reranker'] + '`') if meta['reranker'] else 'non utilisé'}
- Documents ({len(meta['docs'])}) : {', '.join(f'`{d}`' for d in meta['docs'])}
- Questions étiquetées : {meta['n_questions']}
- Taille des chunks en mots ; recouvrement = {meta['overlap_ratio']:.0%} de la taille ; RRF k = {meta['rrf_k']}
- Pertinence binaire : un chunk est pertinent s'il partage une page ou un élément « evidence » avec l'étiquette.
- Latence : encodage de la requête + recherche (+ reranking), indexation exclue, après une requête de chauffe.

{table}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    return table


def update_readme(table: str, readme: Path) -> bool:
    if not readme.exists():
        return False
    txt = readme.read_text(encoding="utf-8")
    pat = re.compile(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)", re.DOTALL)
    if not pat.search(txt):
        return False
    readme.write_text(pat.sub(lambda m: f"{m.group(1)}\n{table}\n{m.group(2)}", txt), encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--questions", default="data/eval/questions.jsonl")
    ap.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    ap.add_argument("--sizes", nargs="+", type=int, default=[128, 256])
    ap.add_argument("--overlap-ratio", type=float, default=0.2)
    ap.add_argument("--model", default=None, help="modele d'embeddings (defaut: multilingual-e5-small)")
    ap.add_argument("--rerank", action="store_true", help="ajoute le mode hybrid+rerank (cross-encoder)")
    ap.add_argument("--rerank-model", default=None)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--readme", default="README.md")
    a = ap.parse_args(argv)

    if len(set(a.sizes)) < 2:
        print("ATTENTION: moins de 2 tailles de chunk demandees (le critere en exige au moins 2).",
              file=sys.stderr)

    # 1. Garde-fou : pas d'etiquettes valides -> erreur claire, AUCUN chiffre.
    corpus = load_corpus(a.processed)
    if not corpus:
        print(f"ERREUR: aucun document extrait dans {a.processed}. Lancez `make download` puis "
              f"`make extract`.", file=sys.stderr)
        return 2
    try:
        questions = load_questions(a.questions, corpus)
    except LabelError as e:
        print(f"ERREUR: {e}", file=sys.stderr)
        return 2

    # 2. Imports lourds seulement apres validation.
    from .index import DEFAULT_EMBEDDING_MODEL, Embedder, EmbedderLoadError, Index
    from .retrieve import DEFAULT_RERANK_MODEL, MODES, Retriever

    try:
        embedder = Embedder(a.model or DEFAULT_EMBEDDING_MODEL)
    except EmbedderLoadError as e:
        print(f"ERREUR: {e}", file=sys.stderr)
        return 3
    modes = [m for m in MODES if a.rerank or m != "hybrid+rerank"]

    rows = []
    for strategy in a.strategies:
        for size in sorted(set(a.sizes)):
            overlap = int(size * a.overlap_ratio)
            chunks = chunk_corpus(corpus, strategy, size, overlap)
            if not chunks:
                print(f"ERREUR: aucun chunk pour {strategy}/{size}.", file=sys.stderr)
                return 2
            relevant_sets = [relevant_chunk_ids(q, chunks, corpus) for q in questions]
            empty = [q.id for q, s in zip(questions, relevant_sets) if not s]
            if empty:
                print(f"ERREUR: {strategy}/{size}: aucun chunk pertinent pour {empty} "
                      f"(etiquette non appariable).", file=sys.stderr)
                return 2
            print(f"[{strategy} size={size} overlap={overlap}] {len(chunks)} chunks", flush=True)
            retriever = Retriever(Index.build(chunks, embedder), embedder,
                                  rerank_model=a.rerank_model, rrf_k=a.rrf_k)
            for mode in modes:
                row = evaluate_config(retriever, questions, relevant_sets, mode)
                rows.append({"strategy": strategy, "size": size, "overlap": overlap, "mode": mode,
                             "n_chunks": len(chunks), "n_questions": len(questions), **row})
                print(f"  {mode:14s} hit@5={row['hit@5']:.3f} mrr={row['mrr']:.3f} "
                      f"ndcg@10={row['ndcg@10']:.3f} lat={row['latency_ms_mean']:.1f}ms", flush=True)

    out = Path(a.out_dir)
    write_csv(rows, out / "results.csv")
    table = write_markdown(rows, out / "results.md", {
        "date": dt.date.today().isoformat(),
        "command": "python -m src.evaluate " + " ".join(argv if argv is not None else sys.argv[1:]),
        "embedder": embedder.model_name,
        "reranker": (a.rerank_model or DEFAULT_RERANK_MODEL) if a.rerank else None,
        "docs": sorted(corpus), "n_questions": len(questions),
        "overlap_ratio": a.overlap_ratio, "rrf_k": a.rrf_k})
    print(f"\n-> {out / 'results.md'} et {out / 'results.csv'}")
    if update_readme(table, Path(a.readme)):
        print(f"-> tableau injecte dans {a.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
