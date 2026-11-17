"""Elastic-net logistic regression, tuned with Optuna and tracked with MLflow.

Run from the repo root: uv run python -m churn.train_logistic
1. base model: every variable, Optuna search on C and l1_ratio by PR-AUC in 5-fold CV on the training set;
2. the same search with 3 engineered features;
3. the engineered features are kept only if they beat the base model by more than one CV standard deviation;
4. the chosen model is refitted on the whole training set and scored once on the hold-out set.
"""

import logging
import warnings

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import cross_validate

from churn.config import ROOT, load_config
from churn.data import load_raw, split
from churn.model import folds, metrics, split_xy
from churn.pipeline import build_pipeline

log = logging.getLogger(__name__)
TEST_METRICS = ["pr_auc", "roc_auc", "precision", "recall"]


def setup_mlflow(cfg: dict) -> None:
    """Local tracking: runs in a SQLite file, models in a folder, both at the repo root whatever the working
    directory (the notebook runs from notebooks/)."""
    ml = cfg["mlflow"]
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / ml['db']).as_posix()}")
    if mlflow.get_experiment_by_name(ml["experiment"]) is None:
        mlflow.create_experiment(ml["experiment"], artifact_location=(ROOT / ml["artifacts"]).as_uri())
    mlflow.set_experiment(ml["experiment"])


def tune(cfg: dict, X: pd.DataFrame, y: pd.Series, fe: bool, run_name: str) -> dict:
    """Optuna (TPE sampler) search on C and l1_ratio by mean CV PR-AUC, under one parent MLflow run; each trial is
    a nested run. Returns the best trial: its parameters, CV mean and CV standard deviation."""
    lr = cfg["logreg"]

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", *lr["C"], log=True)
        l1_ratio = trial.suggest_float("l1_ratio", *lr["l1_ratio"])
        cv = cross_validate(build_pipeline(cfg, C, l1_ratio, fe), X, y, cv=folds(cfg), scoring="average_precision",
                            n_jobs=cfg["model"]["cv_folds"], return_estimator=True)
        mean, std = cv["test_score"].mean(), cv["test_score"].std(ddof=1)
        trial.set_user_attr("cv_pr_auc_std", std)
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params({"C": C, "l1_ratio": l1_ratio})
            mlflow.log_metrics({
                "cv_pr_auc_mean": mean, "cv_pr_auc_std": std,
                "n_features": cv["estimator"][0][-1].n_features_in_,  # columns after encoding
                # Columns the L1 part of the penalty did not set to zero, averaged over the folds.
                "n_nonzero": np.mean([(est[-1].coef_ != 0).sum() for est in cv["estimator"]]),
            })
        return mean

    with mlflow.start_run(run_name=run_name):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
        study.optimize(objective, n_trials=lr["n_trials"])
        best = {**study.best_params, "cv_pr_auc_mean": study.best_value,
                "cv_pr_auc_std": study.best_trial.user_attrs["cv_pr_auc_std"]}
        mlflow.log_params({"C": best["C"], "l1_ratio": best["l1_ratio"]})
        mlflow.log_metrics({"cv_pr_auc_mean": best["cv_pr_auc_mean"], "cv_pr_auc_std": best["cv_pr_auc_std"]})
    return best


def keep_features(base: dict, fe: dict) -> bool:
    """True when the engineered features earn their place: the FE mean beats the base mean by more than one standard
    deviation of the base folds. Otherwise the simpler base model is kept."""
    return fe["cv_pr_auc_mean"] - base["cv_pr_auc_mean"] > base["cv_pr_auc_std"]


def train_final(cfg: dict, train: tuple, test: tuple, best: dict, fe: bool) -> dict:
    """Refit the chosen model on the whole training set, score it once on the hold-out set, log both in MLflow."""
    pipe = build_pipeline(cfg, best["C"], best["l1_ratio"], fe).fit(*train)
    X_test, y_test = test
    scores = metrics(y_test, pipe.predict_proba(X_test)[:, 1], cfg["model"]["threshold"])
    test_scores = {f"test_{k}": scores[k] for k in TEST_METRICS}
    with mlflow.start_run(run_name="logreg_final"):
        mlflow.log_params({"C": best["C"], "l1_ratio": best["l1_ratio"], "feature_engineering": fe,
                           "threshold": cfg["model"]["threshold"]})
        mlflow.log_metrics(test_scores)
        mlflow.log_dict({"features": list(pipe["encode"].get_feature_names_out())}, "features.json")
        # Plain pickle, as the assignment asks: loading it only needs the churn package importable.
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="pickle")
    return test_scores


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Expected: a level too rare in one CV training split shows up in its validation fold and joins the pooled level.
    warnings.filterwarnings("ignore", message="Found unknown categories")
    cfg = load_config()
    train, test = split(load_raw(ROOT / cfg["paths"]["raw_data"]), cfg["target"], cfg["test_size"], cfg["seed"])
    X, y = split_xy(train, cfg)
    setup_mlflow(cfg)

    base = tune(cfg, X, y, fe=False, run_name="logreg_base")
    fe = tune(cfg, X, y, fe=True, run_name="logreg_fe")
    use_fe = keep_features(base, fe)
    # The hold-out set is used here for the first and only time.
    test_scores = train_final(cfg, (X, y), split_xy(test, cfg), fe if use_fe else base, use_fe)

    for name, res in [("base", base), ("FE", fe)]:
        log.info("CV PR-AUC %-4s %.4f +/- %.4f  (C = %.3g, l1_ratio = %.2f)", name, res["cv_pr_auc_mean"],
                 res["cv_pr_auc_std"], res["C"], res["l1_ratio"])
    log.info("Chosen: %s", "FE (beats base by more than 1 SD)" if use_fe else "base (FE gain within 1 SD)")
    log.info("Hold-out: %s", "  ".join(f"{k} {v:.4f}" for k, v in test_scores.items()))


if __name__ == "__main__":
    main()
