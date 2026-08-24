"""Smoke tests for the lightweight models (popularity, item-kNN) that need
only numpy/pandas/scikit-learn -- the heavier torch/surprise/implicit models
are exercised via the full `run-all` pipeline instead of unit tests."""
import pandas as pd
import pytest

from pipeline.data.preprocessing import Dataset
from pipeline.models.base import Recommender
from pipeline.models.knn_cf import ItemKNNRecommender
from pipeline.models.popularity import PopularityRecommender


@pytest.fixture
def toy_dataset():
    # 4 users, 5 items. Users 0 and 1 both love items 0/1 (should look similar);
    # item 3 is rated highly but rarely -> tests popularity vs. quality balance.
    train = pd.DataFrame({
        "user_idx": [0, 0, 1, 1, 2, 2, 3],
        "item_idx": [0, 1, 0, 1, 2, 3, 4],
        "rating": [5, 4, 5, 4, 3, 5, 1],
    })
    test = pd.DataFrame({"user_idx": [0], "item_idx": [2], "rating": [4]})
    books = pd.DataFrame(
        {"title": ["A", "B", "C", "D", "E"], "content_text": ["a", "b", "c", "d", "e"]}
    )
    return Dataset(
        ratings_train=train,
        ratings_test=test,
        books=books,
        n_users=4,
        n_items=5,
        item_id_to_idx={i: i for i in range(5)},
        item_idx_to_id={i: i for i in range(5)},
        user_id_to_idx={i: i for i in range(4)},
        cold_item_idxs={4},
        cold_user_idxs={3},
    )


def test_popularity_scores_all_users_identically(toy_dataset):
    model = PopularityRecommender().fit(toy_dataset)
    assert (model.score(0) == model.score(1)).all()
    assert model.score(0).shape == (5,)


def test_popularity_ranks_frequently_and_highly_rated_items_above_rare_ones(toy_dataset):
    model = PopularityRecommender().fit(toy_dataset)
    scores = model.score(0)
    # item 0 (rated 5,5 by two users) should beat item 4 (rated once, low score)
    assert scores[0] > scores[4]


def test_recommend_all_excludes_seen_items(toy_dataset):
    model = PopularityRecommender().fit(toy_dataset)
    recs = model.recommend_all(toy_dataset, user_idxs=[0], k=5)
    # user 0 has already rated items 0 and 1 in train -> must not reappear
    assert 0 not in recs[0]
    assert 1 not in recs[0]


def test_item_knn_recommends_similar_items_for_users_with_overlapping_taste(toy_dataset):
    model = ItemKNNRecommender(n_neighbors=3).fit(toy_dataset)
    # users 0 and 1 rated exactly the same items (0, 1) -> item 0 and item 1
    # should be each other's nearest neighbor.
    neighbor_ids = [n for n, _sim in model._neighbors[0]]
    assert 1 in neighbor_ids


def test_item_knn_implements_recommender_interface(toy_dataset):
    model = ItemKNNRecommender().fit(toy_dataset)
    assert isinstance(model, Recommender)
    scores = model.score(2)
    assert scores.shape == (5,)
