"""The production models (elastic net, XGBoost) and the steps of the Airflow DAG: prepare the data, train each model,
evaluate it on the hold-out.

Steps hand data to each other as files on the shared volume: each one returns the path it wrote. They need only the
runtime dependencies, so the Airflow image can import this module (mlflow, optuna and matplotlib are development-only).
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from churn.config import ROOT
from churn.data import clean, load_raw, split
from churn.eda import MISSING
from churn.features import add_features, add_features_xgb
from churn.model import _level_name, metrics, split_xy

log = logging.getLogger(__name__)


def build_pipeline(cfg: dict, C: float, l1_ratio: float, fe: bool) -> Pipeline:
    """clean -> (engineered features) -> encode -> elastic-net logistic regression. Medians, scales and levels are
    learned in fit, so on the training rows only, and they are saved with the model."""
    # One unit = one standard deviation, so the penalty treats every variable alike.
    light = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    heavy = make_pipeline(
        # Signed log: compresses the long tails of minutes and revenue, and accepts zeros and negative values.
        FunctionTransformer(np.arcsinh, feature_names_out="one-to-one"),
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    categorical = make_pipeline(
        # A missing value is a level of its own, not the most frequent one: being unknown can carry signal.
        SimpleImputer(strategy="constant", fill_value=MISSING),
        # drop="first": the first level (alphabetical) is the reference, every other level is read against it.
        # Levels below min_frequency are pooled, and levels never seen in training land in that pool.
        OneHotEncoder(drop="first", min_frequency=cfg["model"]["min_frequency"],
                      handle_unknown="infrequent_if_exist", sparse_output=False),
    )
    encode = ColumnTransformer(
        [("light_tailed", light, cfg["logreg"]["light_tailed"]),
         ("categorical", categorical, make_column_selector(dtype_exclude="number"))],
        # Every column not listed above: the heavy-tailed numerics (mou_vs_lifetime included, when added).
        remainder=heavy, verbose_feature_names_out=False)
    steps = [("clean", FunctionTransformer(clean, kw_args={"cfg": cfg["cleaning"]}))]
    if fe:
        # The saved model refers to add_features by its module path: loading it needs only the churn package.
        steps.append(("features", FunctionTransformer(add_features, kw_args={"cfg": cfg})))
    en = cfg["model"]["elastic_net"]
    model = LogisticRegression(C=C, l1_ratio=l1_ratio, solver="saga", max_iter=en["max_iter"], tol=en["tol"],
                               random_state=cfg["seed"])
    return Pipeline([*steps, ("encode", encode), ("model", model)])


def build_xgb_pipeline(cfg: dict, params: dict, formulas: dict[str, str] | None) -> Pipeline:
    """clean -> (engineered features) -> [numbers as they are | one-hot] -> XGBoost. No log, scaling or imputation:
    a split depends only on the order of the values, and XGBoost sends NaN down the branch it learned."""
    steps = [("clean", FunctionTransformer(clean, kw_args={"cfg": cfg["cleaning"]}))]
    if formulas:
        steps.append(("features", FunctionTransformer(add_features_xgb, kw_args={"formulas": formulas})))
    # A missing category becomes a column of its own; a level never seen in training is all zeros. Columns are named
    # "variable=level", so SHAP values can be summed back per variable.
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, feature_name_combiner=_level_name)
    encode = ColumnTransformer([
        ("numeric", "passthrough", make_column_selector(dtype_include="number")),
        ("categorical", onehot, make_column_selector(dtype_exclude="number")),
    ], verbose_feature_names_out=False)
    model = XGBClassifier(**cfg["xgb"]["params"], **params, random_state=cfg["seed"])
    return Pipeline([*steps, ("encode", encode), ("model", model)]).set_output(transform="pandas")


def xgb_formulas(cfg: dict) -> dict[str, str]:
    """The engineered features of the XGBoost model: mou_vs_lifetime + the candidates chosen by the screening."""
    xgb = cfg["xgb"]
    return {**xgb["base_feature"], **{name: xgb["candidates"][name] for name in xgb["chosen"]}}


def _output(cfg: dict, key: str) -> Path:
    path = ROOT / cfg["paths"][key]
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def prepare_data(cfg: dict) -> dict[str, str]:
    """Raw CSV -> stratified train / hold-out split (same config, so the same customers as the notebooks).
    DataFrame pickles keep the dtypes, e.g. the "NA" level of hnd_webcap, without a parquet engine in the image."""
    train, test = split(load_raw(ROOT / cfg["paths"]["raw_data"]), cfg["target"], cfg["test_size"], cfg["seed"])
    paths = {"train": _output(cfg, "train"), "test": _output(cfg, "test")}
    train.to_pickle(paths["train"])
    test.to_pickle(paths["test"])
    return {name: str(path) for name, path in paths.items()}


def _save(pipe: Pipeline, cfg: dict, key: str) -> str:
    """Pickle the whole fitted pipeline (preprocessing included) to the path config gives for key."""
    path = _output(cfg, key)
    with path.open("wb") as f:
        pickle.dump(pipe, f)
    return str(path)


def train_elasticnet(cfg: dict, train_path: str) -> str:
    """Fit the elastic net with the frozen hyperparameters on the training rows; pickle the whole pipeline."""
    X, y = split_xy(pd.read_pickle(train_path), cfg)
    final = cfg["logreg"]["final"]
    return _save(build_pipeline(cfg, final["C"], final["l1_ratio"], final["features"]).fit(X, y), cfg, "model")


def train_xgboost(cfg: dict, train_path: str) -> str:
    """Fit XGBoost with the frozen hyperparameters on the training rows; pickle the whole pipeline."""
    X, y = split_xy(pd.read_pickle(train_path), cfg)
    final = cfg["xgb"]["final"]
    params = {**final["params"], "n_jobs": final["n_jobs"]}
    pipe = build_xgb_pipeline(cfg, params, xgb_formulas(cfg) if final["features"] else None).fit(X, y)
    return _save(pipe, cfg, "xgb_model")


def _describe(model) -> str:
    if isinstance(model, XGBClassifier):
        return (f"XGBoost, {model.n_estimators} trees, max depth {model.max_depth}, "
                f"learning rate {model.learning_rate}")
    return (f"elastic net, C {model.C:.4g}, l1_ratio {model.l1_ratio:.3g}, "
            f"{(model.coef_ != 0).sum()} of {model.coef_.size} columns non-zero")


def evaluate(cfg: dict, model_path: str, train_path: str, test_path: str, threshold: float) -> dict[str, float]:
    """Load the pickled pipeline, score it on the hold-out set, and log the run: data, model, metrics on the
    probabilities, then the decisions at the cut."""
    with open(model_path, "rb") as f:
        pipe = pickle.load(f)
    test = pd.read_pickle(test_path)
    for name, df in [("train", pd.read_pickle(train_path)), ("test", test)]:
        log.info("%-5s %6d rows, churn rate %.3f", name, len(df), df[cfg["target"]].mean())
    # The hyperparameters are read from the pickled model, so the log states what was actually scored.
    log.info("model: %s, engineered features %s", _describe(pipe[-1]), "features" in pipe.named_steps)
    X, y = split_xy(test, cfg)
    proba = pipe.predict_proba(X)[:, 1]
    scores = {**metrics(y, proba, threshold), "brier": brier_score_loss(y, proba)}
    log.info("hold-out metrics on the probabilities (no cut):")
    for name in ["pr_auc", "roc_auc", "brier", "log_loss"]:
        log.info("  %-9s %.4f", name, scores[name])
    log.info("hold-out metrics at the cut, threshold = %.4f (churn if probability >= threshold):", threshold)
    for name in ["precision", "recall", "f1"]:
        log.info("  %-9s %.4f", name, scores[name])
    return scores
