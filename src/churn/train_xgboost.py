"""XGBoost, tuned with Optuna and tracked with MLflow, built like train_logistic.py so the two compare on equal terms.

Run from the repo root: uv run python -m churn.train_xgboost [--screen]
--screen: compare the candidate engineered features on the training set, then stop (chosen ones go in config).
Without it:
1. base model: every variable, Optuna search by PR-AUC in 5-fold CV on the training set (same folds as logreg);
2. the same search with mou_vs_lifetime + the 2 chosen features;
3. the engineered features are kept only if they beat the base model by more than one CV standard deviation;
4. the cut is chosen on out-of-fold training predictions, then the chosen model is refitted on the whole training
   set and scored once on the hold-out set.
"""

import argparse
import logging
import warnings

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, precision_recall_curve
from sklearn.model_selection import cross_val_score, cross_validate

from churn.config import ROOT, load_config
from churn.data import clean, load_raw, split
from churn.features import add_features_xgb
from churn.model import folds, metrics, oof_proba, split_xy
from churn.pipeline import build_xgb_pipeline, xgb_formulas
from churn.plots import plot_pr_curve
from churn.train_logistic import keep_features, setup_mlflow

log = logging.getLogger(__name__)
TEST_METRICS = ["pr_auc", "roc_auc", "brier", "precision", "recall", "f1"]


def cv_pr_auc(cfg: dict, X: pd.DataFrame, y: pd.Series, params: dict, formulas: dict | None) -> np.ndarray:
    """PR-AUC on each of the 5 validation folds (the whole pipeline refitted per fold), folds in parallel."""
    return cross_val_score(build_xgb_pipeline(cfg, params, formulas), X, y, cv=folds(cfg),
                           scoring="average_precision", n_jobs=cfg["model"]["cv_folds"])


def screen(cfg: dict, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each candidate added to mou_vs_lifetime, against mou_vs_lifetime alone, with fixed parameters: the per-fold
    difference in PR-AUC (paired: same folds, so fold-to-fold noise cancels). Also the Spearman correlations of the
    candidates (trees see only the order of a value, so rank correlation is the one that matters)."""
    xgb = cfg["xgb"]
    reference = cv_pr_auc(cfg, X, y, xgb["screen_params"], xgb["base_feature"])
    rows = {}
    for name, formula in xgb["candidates"].items():
        delta = cv_pr_auc(cfg, X, y, xgb["screen_params"], {**xgb["base_feature"], name: formula}) - reference
        rows[name] = {"delta_mean": delta.mean(), "delta_std": delta.std(ddof=1), "folds_won": int((delta > 0).sum())}
    table = pd.DataFrame(rows).T.sort_values("delta_mean", ascending=False)
    formulas = {**xgb["base_feature"], **xgb["candidates"]}
    corr = add_features_xgb(clean(X, cfg["cleaning"]), formulas)[list(formulas)].corr("spearman")
    log.info("Reference (mou_vs_lifetime only): CV PR-AUC %.4f +/- %.4f", reference.mean(), reference.std(ddof=1))
    return table, corr


def tune(cfg: dict, X: pd.DataFrame, y: pd.Series, formulas: dict | None, run_name: str) -> dict:
    """Optuna (TPE sampler) search by mean CV PR-AUC under one parent MLflow run; each trial is a nested run.
    Returns the best trial: its parameters, CV mean and CV standard deviation."""
    space = cfg["xgb"]["search"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
            # Log scale: 1 -> 2 is as big a step as 10 -> 20.
            "min_child_weight": trial.suggest_float("min_child_weight", *space["min_child_weight"], log=True),
            "subsample": trial.suggest_float("subsample", *space["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"], log=True),
        }
        # ROC-AUC rides along at no cost, to compare with earlier models scored by it; the search uses PR-AUC only.
        cv = cross_validate(build_xgb_pipeline(cfg, params, formulas), X, y, cv=folds(cfg),
                            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
                            n_jobs=cfg["model"]["cv_folds"], return_estimator=True)
        mean, std = cv["test_pr_auc"].mean(), cv["test_pr_auc"].std(ddof=1)
        trial.set_user_attr("cv_pr_auc_std", std)
        trial.set_user_attr("cv_scores", cv["test_pr_auc"].tolist())  # per fold, for the paired base-vs-FE delta
        trial.set_user_attr("cv_roc_auc_mean", cv["test_roc_auc"].mean())
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics({"cv_pr_auc_mean": mean, "cv_pr_auc_std": std,
                                "cv_roc_auc_mean": cv["test_roc_auc"].mean(),
                                "n_features": cv["estimator"][0][-1].n_features_in_})  # columns after encoding
        return mean

    with mlflow.start_run(run_name=run_name):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
        study.optimize(objective, n_trials=cfg["xgb"]["n_trials"])
        best = {"params": study.best_params, "cv_pr_auc_mean": study.best_value,
                "cv_pr_auc_std": study.best_trial.user_attrs["cv_pr_auc_std"],
                "cv_scores": np.array(study.best_trial.user_attrs["cv_scores"]),
                "cv_roc_auc_mean": study.best_trial.user_attrs["cv_roc_auc_mean"]}
        mlflow.log_params({**best["params"], "features": list(formulas or [])})
        mlflow.log_metrics({"cv_pr_auc_mean": best["cv_pr_auc_mean"], "cv_pr_auc_std": best["cv_pr_auc_std"],
                            "cv_roc_auc_mean": best["cv_roc_auc_mean"]})
    return best


def recall_threshold(y: pd.Series, proba: np.ndarray, min_precision: float) -> float:
    """The cut with the highest recall among those with precision >= min_precision."""
    precision, _, cuts = precision_recall_curve(y, proba)
    # The last point (recall 0, precision 1) has no cut, so it is left out.
    ok = precision[:-1] >= min_precision
    if not ok.any():
        raise ValueError(f"no cut reaches a precision of {min_precision}")
    # Cuts are ascending and recall can only fall as the cut rises: the lowest acceptable cut has the highest recall.
    return float(cuts[ok][0])


def train_final(cfg: dict, train: tuple, test: tuple, best: dict, formulas: dict | None) -> dict:
    """Choose the cut on out-of-fold training predictions, refit on the whole training set, score once on the
    hold-out set, and log everything in one MLflow run."""
    X, y = train
    pipe = build_xgb_pipeline(cfg, best["params"], formulas)
    oof = oof_proba(pipe, X, y, cfg)  # each training customer scored by a model that did not see it
    threshold = recall_threshold(y, oof, cfg["xgb"]["min_precision"])
    oof_scores = metrics(y, oof, threshold)
    pipe.fit(X, y)
    X_test, y_test = test
    proba = pipe.predict_proba(X_test)[:, 1]
    scores = {**metrics(y_test, proba, threshold), "brier": brier_score_loss(y_test, proba)}
    tn, fp, fn, tp = confusion_matrix(y_test, proba >= threshold).ravel()
    result = {**{f"test_{k}": scores[k] for k in TEST_METRICS}, "test_tn": tn, "test_fp": fp, "test_fn": fn,
              "test_tp": tp, "oof_precision": oof_scores["precision"], "oof_recall": oof_scores["recall"]}
    with mlflow.start_run(run_name="xgb_final"):
        mlflow.log_params({**best["params"], **cfg["xgb"]["params"], "feature_engineering": bool(formulas),
                           "threshold": threshold, "min_precision": cfg["xgb"]["min_precision"]})
        mlflow.log_metrics(result)
        mlflow.log_dict({"features": list(pipe["encode"].get_feature_names_out())}, "features.json")
        mlflow.log_figure(plot_pr_curve(y_test, proba, threshold), "pr_curve_test.png")
        # Plain pickle, as the assignment asks: loading it only needs the churn package and xgboost importable.
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="pickle")
    return {**result, "threshold": threshold}


def logistic_results() -> dict:
    """The final logistic regression's CV and hold-out PR-AUC, read back from its MLflow runs (latest of each)."""
    def latest(name: str) -> pd.Series:
        return mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{name}'", order_by=["start_time DESC"],
                                  max_results=1).iloc[0]

    final = latest("logreg_final")
    cv = latest("logreg_fe" if final["params.feature_engineering"] == "True" else "logreg_base")
    return {"cv_pr_auc_mean": cv["metrics.cv_pr_auc_mean"], "cv_pr_auc_std": cv["metrics.cv_pr_auc_std"],
            "test_pr_auc": final["metrics.test_pr_auc"]}


def report(base: dict, fe: dict, use_fe: bool, final: dict, logistic: dict) -> None:
    for name, res in [("base", base), ("FE", fe)]:
        log.info("CV PR-AUC %-4s %.4f +/- %.4f (ROC-AUC %.4f)  %s", name, res["cv_pr_auc_mean"], res["cv_pr_auc_std"],
                 res["cv_roc_auc_mean"], res["params"])
    # Extra information, not the rule: both best trials ran on the same folds, so their per-fold difference removes
    # the fold-to-fold noise that the rule's standard deviation includes.
    delta = fe["cv_scores"] - base["cv_scores"]
    log.info("Paired delta FE - base: %+.4f +/- %.4f, FE better in %d/%d folds", delta.mean(), delta.std(ddof=1),
             (delta > 0).sum(), len(delta))
    log.info("Chosen: %s", "FE (beats base by more than 1 SD)" if use_fe else "base (FE gain within 1 SD)")
    chosen = fe if use_fe else base
    for name, cv_mean, cv_std, test in [
        ("XGBoost", chosen["cv_pr_auc_mean"], chosen["cv_pr_auc_std"], final["test_pr_auc"]),
        ("logistic", logistic["cv_pr_auc_mean"], logistic["cv_pr_auc_std"], logistic["test_pr_auc"]),
    ]:
        log.info("%-8s CV PR-AUC %.4f +/- %.4f   hold-out PR-AUC %.4f", name, cv_mean, cv_std, test)
    log.info("Cut %.4f (out-of-fold train: precision %.3f, recall %.3f)", final["threshold"],
             final["oof_precision"], final["oof_recall"])
    log.info("Hold-out: %s", "  ".join(f"{k} {final[f'test_{k}']:.4f}" for k in TEST_METRICS))
    log.info("Confusion matrix (rows: true stay/churn, columns: predicted stay/churn): [[%d %d] [%d %d]]",
             final["test_tn"], final["test_fp"], final["test_fn"], final["test_tp"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--screen", action="store_true", help="compare the candidate features, then stop")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Expected: a level too rare to appear in one CV training split shows up in its validation fold (encoded as zeros).
    warnings.filterwarnings("ignore", message="Found unknown categories")
    cfg = load_config()
    train, test = split(load_raw(ROOT / cfg["paths"]["raw_data"]), cfg["target"], cfg["test_size"], cfg["seed"])
    X, y = split_xy(train, cfg)

    if args.screen:
        table, corr = screen(cfg, X, y)
        log.info("Candidates, each added to mou_vs_lifetime (paired CV PR-AUC):\n%s", table.to_string(
            float_format="%.4f"))
        log.info("Spearman correlation:\n%s", corr.to_string(float_format="%.2f"))
        return

    setup_mlflow(cfg)
    formulas = xgb_formulas(cfg)
    base = tune(cfg, X, y, None, run_name="xgb_base")
    fe = tune(cfg, X, y, formulas, run_name="xgb_fe")
    use_fe = keep_features(base, fe)
    # The hold-out set is used here for the first and only time.
    final = train_final(cfg, (X, y), split_xy(test, cfg), fe if use_fe else base, formulas if use_fe else None)
    report(base, fe, use_fe, final, logistic_results())


if __name__ == "__main__":
    main()
