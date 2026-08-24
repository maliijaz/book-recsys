"""Ranking metrics for top-K recommendation evaluation.

All functions operate on a single user's ranked recommendation list against
their set of relevant (held-out) items, plus aggregate helpers that average
over all users and compute catalog-level coverage.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    top_k = recommended[:k]
    return 1.0 if any(item in relevant for item in top_k) else 0.0


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    top_k = recommended[:k]
    dcg = sum(
        1.0 / math.log2(rank + 2) for rank, item in enumerate(top_k) if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def coverage_at_k(recommendations: dict[int, Sequence[int]], n_items: int, k: int) -> float:
    """Fraction of the catalog that appears in at least one user's top-K list."""
    if n_items == 0:
        return 0.0
    recommended_items: set[int] = set()
    for rec_list in recommendations.values():
        recommended_items.update(rec_list[:k])
    return len(recommended_items) / n_items


def evaluate_recommendations(
    recommendations: dict[int, Sequence[int]],
    relevant: dict[int, set[int]],
    k_values: Sequence[int],
    n_items: int,
) -> dict[str, dict[int, float]]:
    """Compute averaged Precision/Recall/NDCG/HitRate@K and Coverage@K.

    `recommendations` and `relevant` are both keyed by user_idx. Only users
    present in `relevant` (i.e. with at least one held-out positive) are
    scored for precision/recall/ndcg/hit-rate.
    """
    results: dict[str, dict[int, float]] = {
        "precision": {}, "recall": {}, "ndcg": {}, "hit_rate": {}, "coverage": {},
    }

    eval_users = [u for u in relevant if u in recommendations]
    for k in k_values:
        if not eval_users:
            results["precision"][k] = 0.0
            results["recall"][k] = 0.0
            results["ndcg"][k] = 0.0
            results["hit_rate"][k] = 0.0
        else:
            results["precision"][k] = sum(
                precision_at_k(recommendations[u], relevant[u], k) for u in eval_users
            ) / len(eval_users)
            results["recall"][k] = sum(
                recall_at_k(recommendations[u], relevant[u], k) for u in eval_users
            ) / len(eval_users)
            results["ndcg"][k] = sum(
                ndcg_at_k(recommendations[u], relevant[u], k) for u in eval_users
            ) / len(eval_users)
            results["hit_rate"][k] = sum(
                hit_rate_at_k(recommendations[u], relevant[u], k) for u in eval_users
            ) / len(eval_users)
        results["coverage"][k] = coverage_at_k(recommendations, n_items, k)

    return results
