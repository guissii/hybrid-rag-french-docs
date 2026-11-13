"""evaluate.py doit refuser de produire des chiffres sans etiquettes valides."""
import json

import pytest

from src import evaluate
from src.chunking import chunk_document
from src.extract import Element


def corpus():
    return {"a.pdf": [Element(0, "title", "Prix", 1, 1),
                      Element(1, "paragraph", "Le prix est ferme et non revisable.", 1),
                      Element(2, "paragraph", "Les delais sont de trente jours.", 2)]}


def write(tmp_path, rows):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def labeled(n, **over):
    return [{"id": f"q{i}", "question": f"question {i} ?",
             "relevant": [{"doc": "a.pdf", "pages": [1]}], **over} for i in range(n)]


def test_missing_and_empty_file(tmp_path):
    with pytest.raises(evaluate.LabelError):
        evaluate.load_questions(tmp_path / "nope.jsonl", corpus())
    with pytest.raises(evaluate.LabelError, match="vide"):
        evaluate.load_questions(write(tmp_path, []), corpus())


def test_template_is_rejected():
    template = [{"id": f"q{i}", "question": "TODO", "relevant": [{"doc": "TODO.pdf", "pages": []}]}
                for i in range(25)]
    import pathlib, tempfile
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(evaluate.LabelError, match="non etiquetees"):
            evaluate.load_questions(write(pathlib.Path(d), template), corpus())


def test_shipped_template_is_rejected():
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "data/eval/questions.jsonl"
    if p.exists():
        txt = p.read_text(encoding="utf-8")
        if "TODO" in txt:
            with pytest.raises(evaluate.LabelError):
                evaluate.load_questions(p, corpus())


def test_too_few_questions(tmp_path):
    with pytest.raises(evaluate.LabelError, match="au moins"):
        evaluate.load_questions(write(tmp_path, labeled(5)), corpus())


def test_unknown_doc_page_and_evidence(tmp_path):
    bad_doc = labeled(20); bad_doc[0]["relevant"] = [{"doc": "x.pdf", "pages": [1]}]
    with pytest.raises(evaluate.LabelError, match="absent"):
        evaluate.load_questions(write(tmp_path, bad_doc), corpus())
    bad_page = labeled(20); bad_page[0]["relevant"] = [{"doc": "a.pdf", "pages": [99]}]
    with pytest.raises(evaluate.LabelError, match="page 99"):
        evaluate.load_questions(write(tmp_path, bad_page), corpus())
    bad_ev = labeled(20); bad_ev[0]["relevant"] = [{"doc": "a.pdf", "evidence": "inexistant"}]
    with pytest.raises(evaluate.LabelError, match="introuvable"):
        evaluate.load_questions(write(tmp_path, bad_ev), corpus())


def test_valid_labels_and_relevance_mapping(tmp_path):
    rows = labeled(20)
    rows[1]["relevant"] = [{"doc": "a.pdf", "evidence": "  LES delais   sont de trente jours. "}]
    qs = evaluate.load_questions(write(tmp_path, rows), corpus())
    assert len(qs) == 20
    chunks = chunk_document(corpus()["a.pdf"], "a.pdf", "fixed", size=6, overlap=0)
    by_page = evaluate.relevant_chunk_ids(qs[0], chunks, corpus())
    by_evidence = evaluate.relevant_chunk_ids(qs[1], chunks, corpus())
    assert by_page and all(1 in chunks[i].pages for i in by_page)
    assert by_evidence and all(2 in chunks[i].elem_ids for i in by_evidence)


def test_main_exits_with_error_and_writes_nothing(tmp_path, capsys):
    import src.extract as ex
    ex.save_elements(corpus()["a.pdf"], tmp_path / "a.pdf.json")
    rc = evaluate.main(["--processed", str(tmp_path), "--questions", str(tmp_path / "none.jsonl"),
                        "--out-dir", str(tmp_path / "res")])
    assert rc == 2 and not (tmp_path / "res").exists()
    assert "ERREUR" in capsys.readouterr().err
