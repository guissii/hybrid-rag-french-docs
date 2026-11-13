"""Extraction DOCX reelle via Docling (ne demande aucun modele a telecharger)."""
import pytest

pytest.importorskip("docling")
docx = pytest.importorskip("docx")

from src.extract import extract_document, load_elements, save_elements


@pytest.fixture
def sample_docx(tmp_path):
    d = docx.Document()
    d.add_heading("Reglement de consultation", level=1)
    d.add_paragraph("Article 1 - Objet de la consultation")
    d.add_paragraph("La presente consultation porte sur des travaux de voirie.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Critere", "Poids"
    t.cell(1, 0).text, t.cell(1, 1).text = "Prix", "60"
    p = tmp_path / "rc.docx"
    d.save(p)
    return p


def test_docx_elements_structure(sample_docx, tmp_path):
    els = extract_document(sample_docx)
    kinds = [e.kind for e in els]
    assert "title" in kinds and "table" in kinds and "paragraph" in kinds
    assert [e.idx for e in els] == list(range(len(els)))
    assert any("voirie" in e.text for e in els)
    assert any(e.kind == "table" and "Prix" in e.text for e in els)
    save_elements(els, tmp_path / "x.json")
    assert load_elements(tmp_path / "x.json") == els


def test_unsupported_format(tmp_path):
    p = tmp_path / "a.txt"; p.write_text("x")
    with pytest.raises(ValueError):
        extract_document(p)
