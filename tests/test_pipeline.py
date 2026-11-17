import numpy as np
import pandas as pd
import pytest

from churn import pipeline


def test_dag_steps_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(5, 2, n)
    raw = tmp_path / "raw.csv"
    pd.DataFrame({
        "id": np.arange(n),
        "x": x,
        "heavy": rng.lognormal(size=n),
        "cat": rng.choice(["A", "B", "U"], n),
        "churn": (x + rng.normal(0, 2, n) > 5).astype(int),
    }).to_csv(raw, sep=";", decimal=",", index=False)
    cfg = {
        # Absolute paths: ROOT / an absolute path is that path.
        "paths": {"raw_data": raw, "train": tmp_path / "p" / "train.pkl", "test": tmp_path / "p" / "test.pkl",
                  "model": tmp_path / "m" / "model.pkl", "xgb_model": tmp_path / "m" / "xgb.pkl"},
        "target": "churn", "exclude": ["id"], "seed": 0, "test_size": 0.25,
        "cleaning": {"unknown_codes": {"cat": ["U"]}, "clip_lower": {}},
        "model": {"threshold": 0.5, "min_frequency": 0.05, "elastic_net": {"max_iter": 1000, "tol": 1e-3}},
        "logreg": {"light_tailed": ["x"], "final": {"C": 1.0, "l1_ratio": 0.5, "features": False}},
        "xgb": {"params": {"learning_rate": 0.1, "tree_method": "hist"},
                "base_feature": {"x_per_heavy": "x / heavy"}, "candidates": {}, "chosen": [],
                "final": {"params": {"n_estimators": 50, "max_depth": 2}, "features": True, "n_jobs": 1}},
    }

    paths = pipeline.prepare_data(cfg)
    model_path = pipeline.train_elasticnet(cfg, paths["train"])
    scores = pipeline.evaluate(cfg, model_path, paths["train"], paths["test"], threshold=0.5, min_pr_auc=0.5)
    xgb_path = pipeline.train_xgboost(cfg, paths["train"])
    xgb_scores = pipeline.evaluate(cfg, xgb_path, paths["train"], paths["test"], threshold=0.5, min_pr_auc=0.5)
    # Quality gate: a model below the minimum PR-AUC fails the task.
    with pytest.raises(ValueError, match="below the minimum"):
        pipeline.evaluate(cfg, model_path, paths["train"], paths["test"], threshold=0.5, min_pr_auc=0.99)

    train, test = pd.read_pickle(paths["train"]), pd.read_pickle(paths["test"])
    assert len(test) == 100 and set(train["id"]).isdisjoint(test["id"])
    assert abs(train["churn"].mean() - test["churn"].mean()) < 0.01  # stratified
    # x drives churn: each pickled pipeline learned it and scores the hold-out.
    assert scores["roc_auc"] > 0.7 and xgb_scores["roc_auc"] > 0.7
    assert 0 < xgb_scores["brier"] < 0.25  # better than a constant 0.5
