import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn import pipeline


def test_dag_steps_round_trip(tmp_path, monkeypatch):
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
    # SQLite stands in for the Postgres business database.
    db = tmp_path / "churn.db"
    monkeypatch.setenv("CHURN_DB_URL", f"sqlite:///{db}")
    cfg = {
        # Absolute paths: ROOT / an absolute path is that path.
        "paths": {"raw_data": raw, "train": tmp_path / "p" / "train.pkl", "test": tmp_path / "p" / "test.pkl",
                  "xgb_candidate": tmp_path / "m" / "candidate" / "xgb.pkl", "xgb_model": tmp_path / "m" / "xgb.pkl"},
        "target": "churn", "id": "id", "exclude": ["id"], "seed": 0, "test_size": 0.25,
        "cleaning": {"unknown_codes": {"cat": ["U"]}, "clip_lower": {}},
        "xgb": {"params": {"learning_rate": 0.1, "tree_method": "hist"},
                "base_feature": {"x_per_heavy": "x / heavy"}, "candidates": {}, "chosen": [],
                "final": {"params": {"n_estimators": 50, "max_depth": 2}, "features": True, "n_jobs": 1,
                          "threshold": 0.5},
                "actions": {"x": "act on x"}, "default_action": "standard"},
    }

    paths = pipeline.prepare_data(cfg)
    candidate = pipeline.train_xgboost(cfg, paths["train"])
    scores = pipeline.evaluate(cfg, candidate, paths["train"], paths["test"], 0.5, min_pr_auc=0.5, run_id="r1")
    model = pipeline.promote(cfg, candidate)
    pipeline.publish_scores(cfg, model, paths["test"], run_id="r1")
    assert not Path(candidate).exists()

    # Next run: a weaker candidate fails the quality gate after its metrics are saved, and is never promoted, so the
    # production model the API and the scores use stays the approved one.
    approved = Path(model).read_bytes()
    cfg["xgb"]["final"]["params"]["n_estimators"] = 5
    candidate = pipeline.train_xgboost(cfg, paths["train"])
    with pytest.raises(ValueError, match="below the minimum"):
        pipeline.evaluate(cfg, candidate, paths["train"], paths["test"], 0.5, min_pr_auc=0.99, run_id="r2")
    assert Path(model).read_bytes() == approved != Path(candidate).read_bytes()

    train, test = pd.read_pickle(paths["train"]), pd.read_pickle(paths["test"])
    assert len(test) == 100 and set(train["id"]).isdisjoint(test["id"])
    assert abs(train["churn"].mean() - test["churn"].mean()) < 0.01  # stratified
    # x drives churn: the pickled pipeline learned it and scores the hold-out.
    assert scores["roc_auc"] > 0.7
    assert 0 < scores["brier"] < 0.25  # better than a constant 0.5

    with sqlite3.connect(db) as conn:
        split = pd.read_sql("SELECT * FROM split", conn)
        history = pd.read_sql("SELECT * FROM model_metrics", conn)
        churn_scores = pd.read_sql("SELECT * FROM churn_scores", conn)
    assert split["dataset"].value_counts().to_dict() == {"train": 300, "test": 100}
    assert history["passed"].tolist() == [1, 0] and history["model"].tolist() == ["xgb", "xgb"]
    assert len(churn_scores) == 100 and set(churn_scores["id"]) == set(test["id"])
    # An action only for flagged customers, and x (the only signal) is the top driver of most of them.
    assert churn_scores["suggested_action"].notna().eq(churn_scores["flagged"] == 1).all()
    assert (churn_scores["suggested_action"] == "act on x").mean() > 0.3