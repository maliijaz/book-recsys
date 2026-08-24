"""Metrics validated against hand-crafted toy rankings with known correct answers."""
import math

from pipeline.evaluation.metrics import (
    coverage_at_k,
    evaluate_recommendations,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_partial_hits():
    recommended = [1, 2, 3, 4, 5]
    relevant = {2, 4, 99}
    assert precision_at_k(recommended, relevant, k=5) == 2 / 5
    assert precision_at_k(recommended, relevant, k=2) == 1 / 2


def test_recall_at_k():
    recommended = [1, 2, 3, 4, 5]
    relevant = {2, 4, 99}
    assert recall_at_k(recommended, relevant, k=5) == 2 / 3
    assert recall_at_k(recommended, relevant, k=1) == 0.0


def test_recall_at_k_no_relevant_items():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


def test_hit_rate_at_k():
    assert hit_rate_at_k([1, 2, 3], {3}, k=3) == 1.0
    assert hit_rate_at_k([1, 2, 3], {3}, k=2) == 0.0
    assert hit_rate_at_k([1, 2, 3], {99}, k=3) == 0.0


def test_ndcg_at_k_perfect_ranking():
    # Both relevant items at the very top -> ideal ranking -> NDCG == 1.
    recommended = [1, 2, 3, 4]
    relevant = {1, 2}
    assert ndcg_at_k(recommended, relevant, k=4) == 1.0


def test_ndcg_at_k_worse_ranking_scores_lower():
    relevant = {1, 2}
    perfect = ndcg_at_k([1, 2, 3, 4], relevant, k=4)
    worse = ndcg_at_k([3, 4, 1, 2], relevant, k=4)
    assert worse < perfect

    # Hand-computed: hit at rank 3 (0-indexed 2) only within k=4 -> dcg = 1/log2(4)
    expected_dcg = 1 / math.log2(2 + 2) + 1 / math.log2(3 + 2)
    expected_idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert math.isclose(worse, expected_dcg / expected_idcg, rel_tol=1e-9)


def test_ndcg_at_k_no_relevant_items_is_zero():
    assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


def test_coverage_at_k():
    recs = {1: [10, 11], 2: [11, 12]}
    # union of top-2 lists = {10, 11, 12} out of a catalog of 6 -> 0.5
    assert coverage_at_k(recs, n_items=6, k=2) == 0.5


def test_evaluate_recommendations_aggregates_across_users():
    recommendations = {1: [10, 11, 12], 2: [20, 21, 22]}
    relevant = {1: {10}, 2: {99}}  # user 1 hits, user 2 misses entirely

    results = evaluate_recommendations(recommendations, relevant, k_values=[3], n_items=100)

    assert results["precision"][3] == (1 / 3 + 0) / 2
    assert results["hit_rate"][3] == 0.5
    assert 0.0 <= results["coverage"][3] <= 1.0


def test_evaluate_recommendations_ignores_users_without_ground_truth():
    recommendations = {1: [10], 2: [20], 3: [30]}
    relevant = {1: {10}}  # only user 1 has a held-out positive

    results = evaluate_recommendations(recommendations, relevant, k_values=[1], n_items=50)

    assert results["hit_rate"][1] == 1.0  # averaged only over user 1, not users 2/3
