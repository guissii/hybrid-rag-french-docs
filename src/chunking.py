"""Trois strategies de decoupage en chunks.

- fixed   : fenetres glissantes de `size` mots avec `overlap` mots de recouvrement,
            sans tenir compte de la structure (baseline).
- section : un chunk par section (delimitee par les titres) ; les sections plus
            longues que `size` sont redecoupees en fenetres. Le chemin de titres
            est prefixe au texte du chunk.
- clause  : un chunk par clause (Article N, 12.3, ...), pour les DOCX
            juridiques/contractuels. Sans marqueur de clause detecte dans le
            document, retombe sur la strategie `section`.

La taille est comptee en MOTS (tokens separes par des espaces), pas en tokens du
modele : c'est une approximation assumee, identique pour toutes les strategies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .extract import Element

STRATEGIES = ("fixed", "section", "clause")

# Mot-cle ("Article 12", "Art. 3", "Clause 4") ou numerotation ("12.3 Obligations",
# "2) Livraison", "4. Prix") suivie d'un mot a majuscule : evite "2024 etait..." ou "1.5 million".
_KEYWORD_CLAUSE_RE = re.compile(r"^\s*(?:article|art\.|clause)\s+[\dIVXivx]+\b", re.IGNORECASE)
_NUMBERED_CLAUSE_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)+[.)]?|\d+[.)])\s+[A-ZÀ-ÝŒ]")


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    text: str
    pages: list[int] = field(default_factory=list)
    elem_ids: list[int] = field(default_factory=list)  # elements sources couverts
    section: str = ""
    strategy: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# (mot, indice de l'element source, page)
_Unit = tuple[str, int, "int | None"]


def _check(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError("size doit etre > 0")
    if not 0 <= overlap < size:
        raise ValueError("overlap doit verifier 0 <= overlap < size")


def _units(elements: list[Element]) -> list[_Unit]:
    return [(w, e.idx, e.page) for e in elements for w in e.text.split()]


def _windows(units: list[_Unit], size: int, overlap: int) -> list[list[_Unit]]:
    if not units:
        return []
    step = size - overlap
    out, start = [], 0
    while True:
        end = min(start + size, len(units))
        out.append(units[start:end])
        if end == len(units):
            return out
        start += step


def _make(doc: str, n: int, units: list[_Unit], strategy: str, section: str = "",
          prefix: str = "", extra_elem_ids: tuple[int, ...] = ()) -> Chunk:
    body = " ".join(u[0] for u in units)
    ids = sorted({u[1] for u in units} | set(extra_elem_ids))
    pages = sorted({u[2] for u in units if u[2] is not None})
    return Chunk(chunk_id=f"{doc}::{strategy}::{n}", doc=doc,
                 text=f"{prefix}\n{body}" if prefix else body,
                 pages=pages, elem_ids=ids, section=section, strategy=strategy)


def chunk_fixed(elements: list[Element], doc: str, size: int = 256, overlap: int = 50) -> list[Chunk]:
    _check(size, overlap)
    title_at: dict[int, str] = {}
    current = ""
    for e in elements:
        if e.kind == "title":
            current = e.text
        title_at[e.idx] = current
    chunks = []
    for n, win in enumerate(_windows(_units(elements), size, overlap)):
        chunks.append(_make(doc, n, win, "fixed", section=title_at.get(win[0][1], "")))
    return chunks


def _split_blocks(elements: list[Element], starts_block) -> list[tuple[list[str], Element | None, list[Element]]]:
    """Regroupe les elements en blocs (chemin_de_titres, element_titre, corps).

    Un nouveau bloc commence a chaque titre, et, si `starts_block(e)` est vrai,
    a chaque element marquant le debut d'une clause.
    """
    blocks: list[tuple[list[str], Element | None, list[Element]]] = []
    path: list[tuple[int, str]] = []
    heading: Element | None = None
    body: list[Element] = []

    def flush():
        if body:
            blocks.append(([t for _, t in path], heading, list(body)))
            body.clear()

    for e in elements:
        if e.kind == "title":
            flush()
            while path and path[-1][0] >= e.level:
                path.pop()
            path.append((e.level, e.text))
            heading = e
        elif starts_block(e):
            flush()
            body.append(e)
        else:
            body.append(e)
    flush()
    return blocks


def _chunk_blocks(blocks, doc: str, strategy: str, size: int, overlap: int) -> list[Chunk]:
    chunks, n = [], 0
    for path, heading, body in blocks:
        prefix = " > ".join(path)
        extra = (heading.idx,) if heading is not None else ()
        for win in _windows(_units(body), size, overlap):
            chunks.append(_make(doc, n, win, strategy, section=path[-1] if path else "",
                                prefix=prefix, extra_elem_ids=extra))
            n += 1
    return chunks


def chunk_by_section(elements: list[Element], doc: str, size: int = 256, overlap: int = 50) -> list[Chunk]:
    _check(size, overlap)
    blocks = _split_blocks(elements, starts_block=lambda e: False)
    return _chunk_blocks(blocks, doc, "section", size, overlap)


def is_clause_start(e: Element) -> bool:
    return e.kind != "table" and bool(_KEYWORD_CLAUSE_RE.match(e.text) or _NUMBERED_CLAUSE_RE.match(e.text))


def chunk_by_clause(elements: list[Element], doc: str, size: int = 256, overlap: int = 50) -> list[Chunk]:
    _check(size, overlap)
    # Un titre "Article 5 - ..." compte aussi comme debut de clause (il ouvre deja un bloc).
    if not any(is_clause_start(e) for e in elements):
        chunks = chunk_by_section(elements, doc, size, overlap)
        for c in chunks:  # tracabilite : la strategie demandee, meme en repli
            c.strategy = "clause"
            c.chunk_id = c.chunk_id.replace("::section::", "::clause::")
        return chunks
    blocks = _split_blocks(elements, starts_block=is_clause_start)
    return _chunk_blocks(blocks, doc, "clause", size, overlap)


_DISPATCH = {"fixed": chunk_fixed, "section": chunk_by_section, "clause": chunk_by_clause}


def chunk_document(elements: list[Element], doc: str, strategy: str,
                   size: int = 256, overlap: int = 50) -> list[Chunk]:
    if strategy not in _DISPATCH:
        raise ValueError(f"Strategie inconnue: {strategy!r} (attendu: {STRATEGIES})")
    return _DISPATCH[strategy](elements, doc, size, overlap)


def chunk_corpus(corpus: dict[str, list[Element]], strategy: str, size: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for doc, els in corpus.items():
        out.extend(chunk_document(els, doc, strategy, size, overlap))
    return out
