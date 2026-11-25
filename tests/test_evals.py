import pytest

from evals import metrics


def test_recall_is_share_of_reference_found():
    assert metrics.recall({"A", "B", "X"}, {"A", "B", "C", "D"}) == 0.5


def test_recall_ignores_found_items_outside_the_reference():
    assert metrics.recall({"X", "Y"}, {"A"}) == 0.0


def test_recall_rejects_empty_reference():
    with pytest.raises(ValueError):
        metrics.recall({"A"}, set())


def test_spearman_perfect_and_reversed():
    assert metrics.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert metrics.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_handles_ties_with_average_ranks():
    # ranks of xs are [1, 2.5, 2.5, 4]; ys are monotone with xs, so the result is still 1.0
    assert metrics.spearman([1, 2, 2, 3], [5, 6, 6, 9]) == pytest.approx(1.0)


def test_spearman_is_none_when_undefined():
    assert metrics.spearman([1], [1]) is None
    assert metrics.spearman([1, 1, 1], [1, 2, 3]) is None   # constant input


def test_spearman_rejects_length_mismatch():
    with pytest.raises(ValueError):
        metrics.spearman([1, 2], [1])


def test_agreement_at_threshold():
    # pairs: (3,3) agree, (1,2) disagree on side of 2, (0,0) agree, (2,1) disagree
    assert metrics.agreement([3, 1, 0, 2], [3, 2, 0, 1], threshold=2) == 0.5


def test_agreement_rejects_empty():
    with pytest.raises(ValueError):
        metrics.agreement([], [], threshold=2)


def test_saturation_counts_the_most_common_score():
    assert metrics.saturation([1.0, 1.0, 1.0, 3.0]) == 0.75
    assert metrics.saturation([1.0, 1.0]) == 1.0
    assert metrics.saturation([0, 1, 2, 3]) == 0.25
