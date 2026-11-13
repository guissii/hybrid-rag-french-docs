import math

import pytest

from src.evaluate import hit_at_k, ndcg_at_k, reciprocal_rank


def test_hit_at_k():
    rels = [False, False, True, False]
    assert hit_at_k(rels, 2) == 0.0 and hit_at_k(rels, 3) == 1.0 and hit_at_k([], 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([True]) == 1.0
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([False] * 5) == 0.0


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k([True, True, False], 10, n_relevant=2) == pytest.approx(1.0)
    assert ndcg_at_k([False] * 10, 10, n_relevant=2) == 0.0
    assert ndcg_at_k([True], 10, n_relevant=0) == 0.0


def test_ndcg_hand_computed():
    # pertinent au rang 2 ; 1 chunk pertinent au total : DCG = 1/log2(3), IDCG = 1
    assert ndcg_at_k([False, True], 10, 1) == pytest.approx(1 / math.log2(3))
    # pertinents aux rangs 1 et 3 sur 3 au total : DCG = 1 + 1/2 ; IDCG = 1 + 1/log2(3) + 1/2
    expected = 1.5 / (1 + 1 / math.log2(3) + 0.5)
    assert ndcg_at_k([True, False, True], 10, 3) == pytest.approx(expected)


def test_ndcg_ideal_capped_by_k():
    # 5 pertinents mais k=2 : l'ideal est 1 + 1/log2(3)
    assert ndcg_at_k([True, True], 2, 5) == pytest.approx(1.0)
