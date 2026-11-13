import pytest

from src.retrieve import rrf_fuse


def test_rrf_scores_match_formula():
    # A: rang 1 + rang 2 ; B: rang 2 + rang 1 ; C: rang 3 dans une seule liste
    res = dict(rrf_fuse([[10, 20, 30], [20, 10]], k=60))
    assert res[10] == pytest.approx(1 / 61 + 1 / 62)
    assert res[20] == pytest.approx(1 / 62 + 1 / 61)
    assert res[30] == pytest.approx(1 / 63)


def test_rrf_consensus_beats_single_list_winner():
    # 2 est 2e partout ; 1 est 1er d'une seule liste -> 2 doit gagner
    fused = [i for i, _ in rrf_fuse([[1, 2, 3], [4, 2, 5]], k=60)]
    assert fused[0] == 2


def test_rrf_order_is_descending_and_deterministic():
    r = rrf_fuse([[3, 1, 2], [2, 3, 1]])
    scores = [s for _, s in r]
    assert scores == sorted(scores, reverse=True)
    assert r == rrf_fuse([[3, 1, 2], [2, 3, 1]])


def test_rrf_tie_broken_by_best_rank_then_id():
    # 7 et 9 ont le meme score (1er d'une liste chacun) -> id le plus petit d'abord
    assert [i for i, _ in rrf_fuse([[9], [7]])] == [7, 9]


def test_rrf_duplicates_in_one_list_counted_once():
    assert dict(rrf_fuse([[1, 1, 1]], k=60))[1] == pytest.approx(1 / 61)


def test_rrf_empty_and_invalid():
    assert rrf_fuse([]) == [] and rrf_fuse([[], []]) == []
    with pytest.raises(ValueError):
        rrf_fuse([[1]], k=-1)
