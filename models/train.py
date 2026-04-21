from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from evaluation.metrics import evaluate_model
from models.baseline.catboost import CatBoostWrapper
from models.baseline.xgboost import XGBoostWrapper
from models.deep.cbm import CBMClassifier
from models.deep.encoder import PairwiseTransformerClassifier, encode_pair_batch


def _generate_sequence(length: int, rng: np.random.Generator) -> str:
    alphabet = np.array(list("ACGT"))
    return "".join(rng.choice(alphabet, size=length).tolist())


def run_baseline_pipeline(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})

    x, y = make_classification(
        n_samples=int(dataset_cfg.get("n_samples", 1200)),
        n_features=int(dataset_cfg.get("n_features", 32)),
        n_informative=int(dataset_cfg.get("n_informative", 10)),
        random_state=int(config.get("seed", 42)),
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=int(config.get("seed", 42)),
        stratify=y,
    )

    xgb = XGBoostWrapper(params=model_cfg.get("xgboost", {})).fit(x_train, y_train)
    cat = CatBoostWrapper(params=model_cfg.get("catboost", {})).fit(x_train, y_train)

    return {
        "xgboost": evaluate_model(xgb, x_test, y_test),
        "catboost": evaluate_model(cat, x_test, y_test),
    }


def run_deep_pipeline(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)

    n_samples = int(config.get("dataset", {}).get("n_samples", 1000))
    seq_len = int(config.get("dataset", {}).get("seq_len", 23))

    guides = [_generate_sequence(seq_len, rng) for _ in range(n_samples)]
    targets = [_generate_sequence(seq_len, rng) for _ in range(n_samples)]
    y = np.asarray([int(g[-3:] == t[-3:]) for g, t in zip(guides, targets)], dtype=float)

    guide_tokens, target_tokens = encode_pair_batch(guides, targets, max_len=seq_len)
    idx = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.25,
        random_state=seed,
        stratify=y.astype(int),
    )

    model = PairwiseTransformerClassifier(
        d_model=int(config.get("model", {}).get("d_model", 32)),
        nhead=int(config.get("model", {}).get("nhead", 4)),
        layers=int(config.get("model", {}).get("layers", 2)),
    )
    model.fit(
        (guide_tokens[train_idx], target_tokens[train_idx]),
        y[train_idx],
        epochs=int(config.get("training", {}).get("epochs", 8)),
        lr=float(config.get("training", {}).get("lr", 1e-3)),
        batch_size=int(config.get("training", {}).get("batch_size", 32)),
    )

    metrics = evaluate_model(
        model,
        (guide_tokens[test_idx], target_tokens[test_idx]),
        y[test_idx],
    )
    return {"transformer_pairwise": metrics}


def run_cbm_pipeline(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    seed = int(config.get("seed", 42))
    dataset_cfg = config.get("dataset", {})

    x, y = make_classification(
        n_samples=int(dataset_cfg.get("n_samples", 1000)),
        n_features=int(dataset_cfg.get("n_features", 20)),
        n_informative=int(dataset_cfg.get("n_informative", 10)),
        random_state=seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )

    model_cfg = config.get("model", {})
    model = CBMClassifier(
        input_dim=x.shape[1],
        concept_dim=int(model_cfg.get("concept_dim", 8)),
        hidden_dim=int(model_cfg.get("hidden_dim", 32)),
    )
    model.fit(
        x_train,
        y_train.astype(float),
        epochs=int(config.get("training", {}).get("epochs", 25)),
        lr=float(config.get("training", {}).get("lr", 1e-3)),
        batch_size=int(config.get("training", {}).get("batch_size", 64)),
    )

    metrics = evaluate_model(model, x_test, y_test)
    return {"cbm": metrics}
