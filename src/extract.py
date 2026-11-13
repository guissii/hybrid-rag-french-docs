"""Extraction layout-aware de PDF et DOCX avec Docling.

Docling convertit un document en un arbre d'elements types (titres, paragraphes,
listes, tableaux) avec leur page d'origine. On aplatit cet arbre en une liste
ordonnee d'`Element` serialisable en JSON, que le reste du pipeline consomme
sans dependre de Docling (le cache permet de ne convertir qu'une fois).

Usage : python -m src.extract data/raw data/processed
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".docx"}

# Etiquettes Docling ignorees : bruit de mise en page qui pollue la recherche.
_SKIPPED_LABELS = {"page_header", "page_footer"}


@dataclass
class Element:
    idx: int                 # position dans le document (identifiant stable)
    kind: str                # "title" | "paragraph" | "list_item" | "table"
    text: str
    page: int | None = None  # None pour les DOCX (pas de pagination fixe)
    level: int = 0           # niveau de titre (0 = titre du document, 1.. = sections)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Element":
        return cls(**d)


def _table_text(item, doc) -> str:
    """Tableau -> texte markdown (compatible avec les versions de docling-core)."""
    try:
        return item.export_to_markdown(doc=doc)
    except TypeError:
        return item.export_to_markdown()


def _page_of(item) -> int | None:
    prov = getattr(item, "prov", None)
    return prov[0].page_no if prov else None


def extract_document(path: str | Path) -> list[Element]:
    """Convertit un PDF/DOCX en liste d'elements structures via Docling."""
    from docling.document_converter import DocumentConverter  # import tardif : lourd
    from docling_core.types.doc import ListItem, SectionHeaderItem, TableItem, TitleItem

    path = Path(path)
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Format non supporte: {path.suffix} (attendu: {sorted(SUPPORTED)})")

    doc = DocumentConverter().convert(str(path)).document
    elements: list[Element] = []
    for item, _depth in doc.iterate_items():
        label = getattr(getattr(item, "label", None), "value", "")
        if label in _SKIPPED_LABELS:
            continue
        if isinstance(item, TableItem):
            kind, text, level = "table", _table_text(item, doc), 0
        else:
            text = (getattr(item, "text", "") or "").strip()
            if isinstance(item, TitleItem):
                kind, level = "title", 0
            elif isinstance(item, SectionHeaderItem):
                kind, level = "title", max(int(getattr(item, "level", 1)), 1)
            elif isinstance(item, ListItem):
                kind, level = "list_item", 0
            else:
                kind, level = "paragraph", 0
        text = text.strip()
        if not text:
            continue
        elements.append(Element(idx=len(elements), kind=kind, text=text,
                                page=_page_of(item), level=level))
    return elements


def save_elements(elements: list[Element], out_path: str | Path) -> None:
    Path(out_path).write_text(
        json.dumps([e.to_dict() for e in elements], ensure_ascii=False, indent=1),
        encoding="utf-8")


def load_elements(path: str | Path) -> list[Element]:
    return [Element.from_dict(d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]


def load_corpus(processed_dir: str | Path) -> dict[str, list[Element]]:
    """Charge tous les documents extraits : {nom_du_fichier_source: elements}."""
    corpus = {}
    for p in sorted(Path(processed_dir).glob("*.json")):
        corpus[p.name[: -len(".json")]] = load_elements(p)  # "x.pdf.json" -> "x.pdf"
    return corpus


def extract_dir(raw_dir: str | Path, out_dir: str | Path, force: bool = False) -> int:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"Aucun PDF/DOCX dans {raw_dir} (lancez `make download` ou deposez des fichiers).",
              file=sys.stderr)
        return 0
    done = 0
    for f in files:
        out = out_dir / f"{f.name}.json"
        if out.exists() and not force and out.stat().st_mtime >= f.stat().st_mtime:
            print(f"= {f.name} (cache)")
            done += 1
            continue
        print(f"> {f.name}")
        try:
            els = extract_document(f)
        except Exception as e:  # un document defectueux ne doit pas tout bloquer
            print(f"  ECHEC: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        save_elements(els, out)
        print(f"  {len(els)} elements")
        done += 1
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw_dir", nargs="?", default="data/raw")
    ap.add_argument("out_dir", nargs="?", default="data/processed")
    ap.add_argument("--force", action="store_true", help="re-extraire meme si le cache est a jour")
    a = ap.parse_args()
    extract_dir(a.raw_dir, a.out_dir, a.force)
