"""CLI entrypoint for the offline ML pipeline.

    python -m pipeline.cli run-all               # preprocess -> train all -> evaluate all -> export
    python -m pipeline.cli preprocess
    python -m pipeline.cli train --model two_tower
    python -m pipeline.cli evaluate --model all
    python -m pipeline.cli export-artifacts
"""
from __future__ import annotations

import argparse
import logging

import joblib

from pipeline.config import ARTIFACTS_DIR, DATA_PROCESSED_DIR
from pipeline.data.download import download_goodbooks
from pipeline.data.preprocessing import Dataset, load_and_split
from pipeline.evaluation.run_eval import evaluate_model
from pipeline.export import artifacts
from pipeline.models.als_bpr import ALSRecommender
from pipeline.models.content_embeddings import ContentEmbeddingRecommender
from pipeline.models.hybrid import HybridRecommender
from pipeline.models.knn_cf import ItemKNNRecommender
from pipeline.models.mf_svd import SVDRecommender
from pipeline.models.ncf import NCFRecommender
from pipeline.models.popularity import PopularityRecommender
from pipeline.models.two_tower import TwoTowerRecommender

logger = logging.getLogger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "dataset.joblib"
MODELS_DIR = DATA_PROCESSED_DIR / "models"

SIMPLE_MODEL_BUILDERS = {
    "popularity": PopularityRecommender,
    "item_knn_cf": ItemKNNRecommender,
    "mf_svd": SVDRecommender,
    "mf_als_implicit": ALSRecommender,
    "content_embeddings": ContentEmbeddingRecommender,
    "ncf": NCFRecommender,
}

# Training order matters: content_embeddings must precede two_tower (which
# consumes its embeddings), which must precede hybrid (which wraps both).
ALL_MODEL_NAMES = [
    "popularity", "item_knn_cf", "mf_svd", "mf_als_implicit",
    "content_embeddings", "ncf", "two_tower", "hybrid",
]


def _load_dataset() -> Dataset:
    if not DATASET_PATH.exists():
        raise SystemExit("No preprocessed dataset found -- run `preprocess` first.")
    return joblib.load(DATASET_PATH)


def _save_model(name: str, model) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / f"{name}.joblib")


def _load_model(name: str):
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise SystemExit(f"No trained model '{name}' found -- run `train --model {name}` first.")
    return joblib.load(path)


def cmd_preprocess(args: argparse.Namespace) -> None:
    download_goodbooks()
    dataset = load_and_split()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dataset, DATASET_PATH)
    logger.info(
        "Preprocessed dataset saved to %s (n_users=%d, n_items=%d, train=%d, test=%d)",
        DATASET_PATH, dataset.n_users, dataset.n_items,
        len(dataset.ratings_train), len(dataset.ratings_test),
    )


def cmd_train(args: argparse.Namespace) -> None:
    dataset = _load_dataset()
    model_name = args.model

    if model_name == "two_tower":
        content_model = _load_model("content_embeddings")
        model = TwoTowerRecommender().fit(dataset, content_model.item_embeddings)
    elif model_name == "hybrid":
        two_tower = _load_model("two_tower")
        content_model = _load_model("content_embeddings")
        model = HybridRecommender(two_tower, content_model).fit(dataset)
    elif model_name in SIMPLE_MODEL_BUILDERS:
        model = SIMPLE_MODEL_BUILDERS[model_name]().fit(dataset)
    else:
        raise SystemExit(f"Unknown model '{model_name}'")

    _save_model(model_name, model)
    logger.info("Trained and saved model '%s'", model_name)


def cmd_evaluate(args: argparse.Namespace) -> list[dict]:
    dataset = _load_dataset()
    names = ALL_MODEL_NAMES if args.model == "all" else [args.model]
    results = []
    for name in names:
        model = _load_model(name)
        result = evaluate_model(model, dataset)
        results.append(result)
        logger.info(
            "Evaluated %-20s NDCG@10=%.4f Recall@10=%.4f ColdUserNDCG@10=%.4f",
            name,
            result["overall"]["ndcg"][10],
            result["overall"]["recall"][10],
            result["cold_start"]["cold_user_metrics"]["ndcg"][10],
        )
    artifacts.export_metrics(results)
    return results


def cmd_export_artifacts(args: argparse.Namespace) -> None:
    dataset = _load_dataset()
    two_tower = _load_model("two_tower")
    content_model = _load_model("content_embeddings")
    hybrid = _load_model("hybrid")

    artifacts.export_book_metadata(dataset)
    artifacts.export_embeddings(two_tower.item_embeddings, content_model.item_embeddings)
    artifacts.export_cf_weight(hybrid.cf_weight)
    artifacts.export_personas(dataset, hybrid)
    logger.info("Artifacts exported to %s", ARTIFACTS_DIR)


def cmd_run_all(args: argparse.Namespace) -> None:
    cmd_preprocess(args)
    for name in ALL_MODEL_NAMES:
        cmd_train(argparse.Namespace(model=name))
    cmd_evaluate(argparse.Namespace(model="all"))
    cmd_export_artifacts(args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Book recommender offline ML pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preprocess").set_defaults(func=cmd_preprocess)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", required=True, choices=ALL_MODEL_NAMES)
    train_parser.set_defaults(func=cmd_train)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--model", default="all", choices=[*ALL_MODEL_NAMES, "all"])
    eval_parser.set_defaults(func=cmd_evaluate)

    subparsers.add_parser("export-artifacts").set_defaults(func=cmd_export_artifacts)
    subparsers.add_parser("run-all").set_defaults(func=cmd_run_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
