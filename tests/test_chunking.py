import pytest

from src.chunking import (chunk_by_clause, chunk_by_section, chunk_document, chunk_fixed,
                          is_clause_start)
from src.extract import Element


def els(*specs):
    out = []
    for i, s in enumerate(specs):
        kind, text, *rest = s
        out.append(Element(idx=i, kind=kind, text=text,
                           page=rest[0] if rest else None, level=rest[1] if len(rest) > 1 else 0))
    return out


def words(n, prefix="m"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_fixed_window_sizes_and_overlap():
    chunks = chunk_fixed(els(("paragraph", words(25), 1)), "d", size=10, overlap=3)
    sizes = [len(c.text.split()) for c in chunks]
    assert sizes == [10, 10, 10, 4]  # pas de 7 : debuts 0,7,14,21 ; la queue couvre 25-21 mots
    assert chunks[0].text.split()[7:] == chunks[1].text.split()[:3]  # recouvrement de 3 mots


def test_fixed_covers_every_word_and_no_empty_chunk():
    text = words(53)
    chunks = chunk_fixed(els(("paragraph", text, 1)), "d", size=8, overlap=2)
    assert all(c.text for c in chunks)
    assert set(" ".join(c.text for c in chunks).split()) == set(text.split())


def test_fixed_exact_multiple_has_no_trailing_duplicate():
    chunks = chunk_fixed(els(("paragraph", words(10), 1)), "d", size=10, overlap=3)
    assert len(chunks) == 1


@pytest.mark.parametrize("size,overlap", [(0, 0), (5, 5), (5, 9), (5, -1)])
def test_invalid_params(size, overlap):
    for fn in (chunk_fixed, chunk_by_section, chunk_by_clause):
        with pytest.raises(ValueError):
            fn(els(("paragraph", "a b c")), "d", size=size, overlap=overlap)


def test_fixed_tracks_pages_across_elements():
    chunks = chunk_fixed(els(("paragraph", words(6), 1), ("paragraph", words(6, "z"), 2)),
                         "d", size=8, overlap=0)
    assert chunks[0].pages == [1, 2]  # fenetre a cheval sur deux pages
    assert chunks[1].pages == [2]


def test_section_splits_on_titles_and_prefixes_path():
    e = els(("title", "Contrat", 1, 1), ("paragraph", "intro un", 1),
            ("title", "Prix", 2, 2), ("paragraph", "le prix est ferme", 2),
            ("title", "Delais", 3, 1), ("paragraph", "trente jours", 3))
    chunks = chunk_by_section(e, "d", size=50, overlap=5)
    assert len(chunks) == 3
    assert chunks[1].text.startswith("Contrat > Prix\n")  # chemin hierarchique
    assert chunks[2].text.startswith("Delais\n")           # remonte au niveau 1
    assert chunks[1].section == "Prix"
    assert 2 in chunks[1].elem_ids and chunks[1].pages == [2]  # l'element titre est couvert


def test_section_long_body_is_windowed_with_prefix():
    e = els(("title", "Annexe", 1, 1), ("paragraph", words(30), 1))
    chunks = chunk_by_section(e, "d", size=10, overlap=2)
    assert len(chunks) > 1 and all(c.text.startswith("Annexe\n") for c in chunks)
    assert all(len(c.text.split("\n", 1)[1].split()) <= 10 for c in chunks)


def test_title_without_body_makes_no_chunk():
    assert chunk_by_section(els(("title", "Vide", 1, 1)), "d") == []


@pytest.mark.parametrize("text,expected", [
    ("Article 12 - Penalites", True), ("art. 3 delais", True), ("Clause 4 : prix", True),
    ("12.3 Obligations du titulaire", True), ("2) Livraison", True),
    ("Le prix est de 12 euros", False), ("1.5 million d'euros", False), ("2024 etait une annee", False), ("12.5", False),
])
def test_clause_detection(text, expected):
    assert is_clause_start(Element(0, "paragraph", text)) is expected


def test_clause_chunks_follow_clauses():
    e = els(("title", "CCAP", 1, 1),
            ("paragraph", "Article 1 Objet", 1), ("paragraph", "Le marche porte sur X.", 1),
            ("paragraph", "Article 2 Prix", 1), ("paragraph", "Le prix est ferme.", 1))
    chunks = chunk_by_clause(e, "d", size=50, overlap=5)
    assert len(chunks) == 2
    assert "Article 1 Objet" in chunks[0].text and "Article 2" not in chunks[0].text
    assert "Le prix est ferme." in chunks[1].text
    assert all(c.strategy == "clause" for c in chunks)


def test_clause_falls_back_to_section_without_markers():
    e = els(("title", "Note", 1, 1), ("paragraph", "texte libre sans numerotation", 1))
    chunks = chunk_by_clause(e, "d", size=50, overlap=5)
    assert len(chunks) == 1 and chunks[0].strategy == "clause" and "::clause::" in chunks[0].chunk_id


def test_dispatch_and_unique_ids():
    e = els(("paragraph", words(40), 1))
    chunks = chunk_document(e, "d", "fixed", 10, 2)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    with pytest.raises(ValueError):
        chunk_document(e, "d", "inconnue")
