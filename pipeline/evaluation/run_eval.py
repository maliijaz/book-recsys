"""Evaluate a fitted recommender against the held-out test split."""
from __future__ import annotations

from pipeline.config import TOP_K_VALUES
from pipeline.data.preprocessing import Dataset
from pipeline.evaluation.cold_start_slice import evaluate_cold_start
from pipeline.evaluation.metrics import evaluate_recommendations
from pipeline.models.base import Recommender


def evaluate_model(model: Recommender, dataset: Dataset, k_values=TOP_K_VALUES) -> dict:
    relevant = dataset.ratings_test.groupby("user_idx")["item_idx"].apply(set).to_dict()
    max_k = max(k_values)

    recommendations = model.recommend_all(dataset, list(relevant.keys()), k=max_k)

    overall = evaluate_recommendations(recommendations, relevant, k_values, dataset.n_items)
    cold_start = evaluate_cold_start(recommendations, relevant, dataset, k_values)

    return {
        "model": model.name,
        "overall": overall,
        "cold_start": cold_start,
    }
