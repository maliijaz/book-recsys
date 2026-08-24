"""Cold-start evaluation slice: how well a model serves low-interaction users/items.

Pure collaborative filtering models tend to collapse toward zero on this
slice because they have little-to-no signal for cold users/items; the
content-based and hybrid models are expected to recover performance here.
"""
from __future__ import annotations

from collections.abc import Sequence

from pipeline.data.preprocessing import Dataset
from pipeline.evaluation.metrics import evaluate_recommendations


def evaluate_cold_start(
    recommendations: dict[int, Sequence[int]],
    relevant: dict[int, set[int]],
    dataset: Dataset,
    k_values: Sequence[int],
) -> dict:
    """Split the standard evaluation into a cold-user slice and a cold-item recall metric."""
    cold_relevant = {
        u: items for u, items in relevant.items() if u in dataset.cold_user_idxs
    }
    cold_user_metrics = evaluate_recommendations(
        recommendations, cold_relevant, k_values, dataset.n_items
    )

    # Cold-item recall@k: of all held-out test positives that are cold items,
    # what fraction actually appear in their user's top-k recommendations?
    cold_item_recall: dict[int, float] = {}
    for k in k_values:
        total_cold_positives = 0
        recovered = 0
        for user_idx, items in relevant.items():
            cold_positives = items & dataset.cold_item_idxs
            if not cold_positives:
                continue
            total_cold_positives += len(cold_positives)
            top_k = set(recommendations.get(user_idx, [])[:k])
            recovered += len(cold_positives & top_k)
        cold_item_recall[k] = recovered / total_cold_positives if total_cold_positives else 0.0

    return {
        "cold_user_metrics": cold_user_metrics,
        "cold_item_recall": cold_item_recall,
        "n_cold_users": len(dataset.cold_user_idxs),
        "n_cold_items": len(dataset.cold_item_idxs),
    }
