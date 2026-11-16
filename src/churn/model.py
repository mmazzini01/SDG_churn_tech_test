"""Logistic regression, GAM (splines + logistic) and XGBoost pipelines, their tuning by CV, the decision threshold,
the metrics shared by CV and the hold-out set, and the per-variable explanations (exact SHAP for all three)."""

import re
from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    SplineTransformer,
    StandardScaler,
)
from xgboost import DMatrix, XGBClassifier

from churn.data import clean
from churn.eda import MISSING, assign_families
from churn.features import add_ratios

INDICATOR = "missingindicator_"
# Spline columns are "<variable>_sp_<k>" (SplineTransformer), tensor-product columns "<a> × <b>_tp_<i>_<j>".
TERM = re.compile(r"_(?:sp_\d+|tp_\d+_\d+)$")


def split_xy(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Features and target; the key and the excluded columns never reach the model."""
    return df.drop(columns=[cfg["target"], *cfg["exclude"]]), df[cfg["target"]]


def _level_name(feature: str, category: str) -> str:
    # "variable=level" lets every one-hot column be traced back to its variable (see variable_of).
    return f"{feature}={category}"


def _prepare(cfg: dict, ratios: bool, trend: bool) -> list[tuple]:
    """The steps both model families share: the conformity cleaning, then the engineered ratios: the scale-free
    ratios, the trend ratios, both or none."""
    chosen = {**(cfg["features"]["ratios"] if ratios else {}), **(cfg["features"].get("trend", {}) if trend else {})}
    add = FunctionTransformer(add_ratios, kw_args={"ratios": chosen}) if chosen else "passthrough"
    return [("clean", FunctionTransformer(clean, kw_args={"cfg": cfg["cleaning"]})), ("ratios", add)]


def _categorical(cfg: dict) -> Pipeline:
    # A missing category is a level of its own, never the mode, so the recoded unknown codes keep their signal.
    return make_pipeline(
        SimpleImputer(strategy="constant", fill_value=MISSING),
        OneHotEncoder(min_frequency=cfg["model"]["min_frequency"], handle_unknown="infrequent_if_exist",
                      sparse_output=False, feature_name_combiner=_level_name))


def _numeric(cfg: dict, n_knots: int | None) -> list:
    # arcsinh is a signed log: long tails compressed, zeros and negative values (credits, differences) allowed.
    steps = [FunctionTransformer(np.arcsinh, feature_names_out="one-to-one"), SimpleImputer(strategy="median")]
    if n_knots:
        # Quantile knots put the same number of customers between knots, so no curve piece rests on a few outliers;
        # past the training range the curve stays flat instead of extrapolating a cubic.
        steps.append(SplineTransformer(n_knots=n_knots, degree=cfg["model"]["gam"]["degree"], knots="quantile",
                                       extrapolation="constant", include_bias=False))
    return steps


def _tensor(B: np.ndarray) -> np.ndarray:
    """Each basis column of the first variable times each of the second: the pair's tensor-product basis, whose
    weighted sum can draw any smooth surface over the two values."""
    a, b = np.hsplit(np.asarray(B), 2)
    return (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)


def _tensor_names(_, input_features: np.ndarray) -> list[str]:
    half = len(input_features) // 2
    term = f"{variable_of(input_features[0])} × {variable_of(input_features[half])}"
    return [f"{term}_tp_{i}_{j}" for i in range(half) for j in range(half)]


def build_pipeline(cfg: dict, estimator: ClassifierMixin, ratios: bool = True, trend: bool = True,
                   n_knots: int | None = None, tensors: list[list[str]] | None = None) -> Pipeline:
    """clean -> ratios -> [signed log + median (+ splines) | tensor products | missing-block indicators | one-hot with
    a Missing level] -> scale -> model. With n_knots every numeric becomes a spline basis, so the model is a GAM;
    each pair in tensors adds a surface over its two variables on top of their curves. Every statistic is learned in
    fit, so the pickle carries the training set's medians, knots and scales."""
    pairs = [(f"tensor_{a}_{b}", make_pipeline(*_numeric(cfg, cfg["model"]["gam"]["tensor_knots"]),
                                                FunctionTransformer(_tensor, feature_names_out=_tensor_names)), [a, b])
             for a, b in tensors or []]
    encode = ColumnTransformer([
        ("numeric", make_pipeline(*_numeric(cfg, n_knots)), make_column_selector(dtype_include="number")),
        *pairs,
        ("missing", MissingIndicator(features="all"), cfg["model"]["missing_indicators"]),
        ("categorical", _categorical(cfg), make_column_selector(dtype_exclude="number")),
    ], verbose_feature_names_out=False)
    return Pipeline([
        *_prepare(cfg, ratios, trend),
        ("encode", encode),
        # Same scale for every column, dummies included, so the penalty does not depend on a column's units.
        ("scale", StandardScaler()),
        ("model", estimator),
    ]).set_output(transform="pandas")


def build_tree_pipeline(cfg: dict, estimator: ClassifierMixin, ratios: bool = True, trend: bool = True) -> Pipeline:
    """clean -> ratios -> [numbers as they are, NaN kept | one-hot with a Missing level] -> trees. No log, scaling
    or imputation: a split depends only on the order of the values, and XGBoost sends NaN down a branch it learns."""
    encode = ColumnTransformer([
        ("numeric", "passthrough", make_column_selector(dtype_include="number")),
        ("categorical", _categorical(cfg), make_column_selector(dtype_exclude="number")),
    ], verbose_feature_names_out=False)
    return Pipeline([*_prepare(cfg, ratios, trend), ("encode", encode), ("model", estimator)]).set_output(
        transform="pandas")


def folds(cfg: dict) -> StratifiedKFold:
    """The one set of folds used for tuning and for comparing models."""
    return StratifiedKFold(cfg["model"]["cv_folds"], shuffle=True, random_state=cfg["seed"])


def baseline(cfg: dict) -> LogisticRegression:
    return LogisticRegression(**cfg["model"]["baseline"])


def elastic_net_cv(cfg: dict) -> LogisticRegressionCV:
    """C path x l1_ratios on the CV folds. Each path is warm-started from the previous C, so the whole grid costs
    about one fit per fold and l1_ratio. refit=False: the final model is the best row of cv_path."""
    en = cfg["model"]["elastic_net"]
    return LogisticRegressionCV(
        # An array, not a list: with refit=False sklearn 1.9 indexes l1_ratios with an array of fold winners.
        Cs=np.logspace(*en["log10_C"], en["n_Cs"]), l1_ratios=np.array(en["l1_ratios"]), solver="saga", cv=folds(cfg),
        scoring=en["scoring"], max_iter=en["max_iter"], tol=en["tol"], refit=False, n_jobs=-1,
        random_state=cfg["seed"], use_legacy_attributes=False)


def elastic_net(cfg: dict, C: float, l1_ratio: float) -> LogisticRegression:
    en = cfg["model"]["elastic_net"]
    return LogisticRegression(C=C, l1_ratio=l1_ratio, solver="saga", max_iter=en["max_iter"], tol=en["tol"],
                              random_state=cfg["seed"])


def gam_cv(cfg: dict) -> LogisticRegressionCV:
    """Ridge path over C on the CV folds, for one spline design; the final model is chosen by pick_one_se."""
    g = cfg["model"]["gam"]
    return LogisticRegressionCV(Cs=np.logspace(*g["log10_C"], g["n_Cs"]), l1_ratios=np.array([0.0]), cv=folds(cfg),
                                scoring=g["scoring"], max_iter=g["max_iter"], refit=False, n_jobs=g["n_jobs"],
                                use_legacy_attributes=False)


def gam(cfg: dict, C: float) -> LogisticRegression:
    # Ridge (sklearn's default penalty): it shrinks the spline coefficients, so a smaller C draws smoother curves.
    return LogisticRegression(C=C, max_iter=cfg["model"]["gam"]["max_iter"])


def xgboost(cfg: dict, max_depth: int, n_trees: int, **tuned: float) -> XGBClassifier:
    """The fixed settings of config (tree method, threads) plus the given hyperparameters."""
    return XGBClassifier(max_depth=max_depth, n_estimators=n_trees, random_state=cfg["seed"],
                         **{**cfg["model"]["xgboost"]["params"], **tuned})


def tune_xgboost(cfg: dict, X: pd.DataFrame, y: pd.Series):
    """Bayesian search (Optuna's TPE sampler) over the ranges in config: each trial scores the whole tree pipeline by
    CV log-loss on the folds, and the sampler proposes the next values from the trials so far. Returns the study."""
    import optuna  # dev dependency: only the notebook tunes, the pipeline reads the frozen values from config

    xg = cfg["model"]["xgboost"]

    def cv_loss(trial: optuna.Trial) -> float:
        # Integer bounds give integer values; the learning rate is drawn on a log scale (0.01 -> 0.02 is as big a
        # step as 0.1 -> 0.2).
        params = {name: trial.suggest_int(name, lo, hi) if isinstance(lo, int)
                  else trial.suggest_float(name, lo, hi, log=name == "learning_rate")
                  for name, (lo, hi) in xg["search"].items()}
        cv = cross_validate(build_tree_pipeline(cfg, xgboost(cfg, **params)), X, y, cv=folds(cfg),
                            scoring="neg_log_loss", n_jobs=cfg["model"]["cv_folds"])
        return -cv["test_score"].mean()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.optimize(cv_loss, n_trials=xg["n_trials"])
    return study


def cv_path(lrcv: LogisticRegressionCV) -> pd.DataFrame:
    """Per (l1_ratio, C): mean and standard error of the CV score over the folds, and the mean number of
    non-zero coefficients."""
    scores = lrcv.scores_  # (folds, l1_ratios, Cs)
    # (folds, l1_ratios, Cs, 1, features + intercept): the intercept is last and is never penalised.
    nonzero = (lrcv.coefs_paths_[..., 0, :-1] != 0).sum(axis=-1)
    grid = pd.MultiIndex.from_product([lrcv.l1_ratios_, lrcv.Cs_], names=["l1_ratio", "C"])
    return pd.DataFrame({
        "score": scores.mean(axis=0).ravel(),
        "se": scores.std(axis=0, ddof=1).ravel() / np.sqrt(len(scores)),
        "n_nonzero": nonzero.mean(axis=0).ravel(),
    }, index=grid).reset_index()


def pick_one_se(path: pd.DataFrame) -> pd.Series:
    """One-standard-error rule: among the configurations whose mean score is within one SE of the best, the one
    with the fewest non-zero coefficients (then the smallest C): a sparser model whose score is within the
    fold-to-fold noise of the best. That SE is unpaired, so a paired comparison on the same folds can still detect
    a small loss."""
    best = path.loc[path["score"].idxmax()]
    close = path[path["score"] >= best["score"] - best["se"]]
    return close.sort_values(["n_nonzero", "C"]).iloc[0]


def metrics(y: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    """Ranking metrics on the probabilities; precision, recall and F1 on the decisions at the threshold."""
    pred = proba >= threshold
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "log_loss": log_loss(y, proba),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
    }


def _score(estimator: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float) -> dict[str, float]:
    return metrics(y, estimator.predict_proba(X)[:, 1], threshold)


def cv_metrics(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cfg: dict, n_jobs: int | None = None) -> pd.DataFrame:
    """metrics() on each validation fold; the whole pipeline, preprocessing included, is refitted per fold. The folds
    run in parallel, all at once unless n_jobs caps them (memory)."""
    scorer = partial(_score, threshold=cfg["model"]["threshold"])
    res = cross_validate(pipe, X, y, cv=folds(cfg), scoring=scorer, n_jobs=n_jobs or cfg["model"]["cv_folds"])
    return pd.DataFrame({k.removeprefix("test_"): v for k, v in res.items() if k.startswith("test_")})


def oof_proba(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cfg: dict) -> np.ndarray:
    """Out-of-fold probabilities: each training customer scored by the pipeline fitted on the other folds."""
    return cross_val_predict(pipe, X, y, cv=folds(cfg), method="predict_proba", n_jobs=cfg["model"]["cv_folds"])[:, 1]


def best_f1_threshold(y: pd.Series, proba: np.ndarray) -> float:
    """The probability cut with the highest F1 among every distinct cut (precision_recall_curve)."""
    precision, recall, cuts = precision_recall_curve(y, proba)
    # The last point (recall 0, precision 1) has no cut, so it is left out.
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(cuts[np.argmax(f1)])


def variable_of(column: str) -> str:
    """Input variable of an encoded column: 'crclscod=A' -> 'crclscod', 'missingindicator_x' -> 'x (missing)',
    'eqpdays_sp_3' -> 'eqpdays', 'eqpdays × months_tp_1_2' -> 'eqpdays × months' (a term of its own)."""
    if column.startswith(INDICATOR):
        return column.removeprefix(INDICATOR) + " (missing)"
    return TERM.sub("", column.split("=")[0])


def coefficients(pipe: Pipeline) -> pd.DataFrame:
    """Per encoded column: coefficient per standard deviation, its odds ratio, and the odds ratio per unit, which
    is the readable one for 0/1 columns (a level, a missing indicator)."""
    scale = pipe["scale"]
    coef = pd.Series(pipe[-1].coef_.ravel(), index=scale.get_feature_names_out())
    return pd.DataFrame({"variable": coef.index.map(variable_of), "coef": coef, "or_per_sd": np.exp(coef),
                         "or_per_unit": np.exp(coef / scale.scale_)})


def contributions(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Log-odds contribution of each input variable for each customer, against the training-set average: the
    coefficients times the standardised values, summed over the variable's columns. For a model that is additive in
    its variables (logistic regression, GAM) this is exactly its SHAP value, so the same table gives the global
    importance and the single-customer explanation."""
    z = pipe[:-1].transform(X)
    coef = pd.Series(pipe[-1].coef_.ravel(), index=z.columns)
    return (z * coef).T.groupby(variable_of).sum().T


def fold_contributions(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, X_eval: pd.DataFrame, cfg: dict) -> list:
    """contributions() on X_eval of the pipeline refitted on each CV training split: how much each variable's curve
    moves when the training customers change."""
    return [contributions(clone(pipe).fit(X.iloc[tr], y.iloc[tr]), X_eval) for tr, _ in folds(cfg).split(X, y)]


def joint_effect(pipe: Pipeline, row: pd.DataFrame, x: str, x_grid: list, line: str, line_grid: list) -> pd.DataFrame:
    """Log-odds contribution of a pair of variables over a grid of their values: both curves plus the pair's tensor
    term, if the model has one. The model is additive, so the customer in `row` (one row of X) only fills the other
    columns and does not change the result. One row per value of `line`, one column per value of `x`."""
    grid = pd.MultiIndex.from_product([line_grid, x_grid], names=[line, x]).to_frame(index=False)
    X = row.loc[row.index.repeat(len(grid))].reset_index(drop=True).assign(**{x: grid[x], line: grid[line]})
    contrib = contributions(pipe, X)
    terms = [t for t in (x, line, f"{x} × {line}") if t in contrib]
    return contrib[terms].sum(axis=1).set_axis(pd.MultiIndex.from_frame(grid)).unstack(x)


def on_support(joint: pd.DataFrame, X: pd.DataFrame, n: int) -> pd.DataFrame:
    """joint_effect() with the cells where there are no customers blanked: on each line, x is kept between the 1st
    and 99th percentile of the n training customers whose line value is closest (all of them when more are tied, so
    the row order never matters). Each variable can be inside its own range while the pair is not (a 330-day handset
    at 8 months of tenure), and a surface there is extrapolation."""
    x, line = joint.columns.name, joint.index.name
    out = joint.copy()
    for level in joint.index:
        dist = (X[line] - level).abs()
        lo, hi = X.loc[dist <= dist.nsmallest(n).iloc[-1], x].quantile([0.01, 0.99])
        out.loc[level, (out.columns < lo) | (out.columns > hi)] = np.nan
    return out


def importance(pipe: Pipeline, X: pd.DataFrame, families: dict[str, list[str]]) -> pd.DataFrame:
    """Per input variable: mean |contribution| (log-odds), whether any of its columns survived the penalty, and
    the direction for single-column variables ('+': higher value, or missing, goes with more churn)."""
    coefs = coefficients(pipe).groupby("variable")["coef"]
    single = coefs.size() == 1
    sign = np.sign(coefs.first())
    out = pd.DataFrame({
        "importance": contributions(pipe, X).abs().mean(),
        "selected": coefs.apply(lambda c: (c != 0).any()),
        "direction": pd.Series(np.where(single & (sign > 0), "+", np.where(single & (sign < 0), "-", "")),
                               index=single.index),
    })
    out.insert(0, "family", assign_families(list(out.index), families))
    return out.sort_values("importance", ascending=False).rename_axis("variable")


def tree_contributions(pipe: Pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Exact TreeSHAP values (log-odds) from XGBoost's own implementation, one-hot columns summed per input variable,
    and the base value they start from: base + a customer's contributions = the model's log-odds for that customer."""
    Xt = pipe[:-1].transform(X)
    shap = pipe[-1].get_booster().predict(DMatrix(Xt), pred_contribs=True)
    # The last column is the bias: the same expected log-odds for every customer.
    contrib = pd.DataFrame(shap[:, :-1], index=X.index, columns=Xt.columns).T.groupby(variable_of).sum().T
    return contrib, float(shap[0, -1])


def tree_interactions(pipe: Pipeline, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP interaction value| per pair of input variables (XGBoost's exact TreeSHAP), one-hot columns summed
    into their variable, strongest pair first. SHAP splits a pair's interaction equally between (a, b) and (b, a), so
    a pair gets twice one cell. Memory grows with rows x columns², so X should be a sample of a few thousand rows."""
    Xt = pipe[:-1].transform(X)
    inter = pipe[-1].get_booster().predict(DMatrix(Xt), pred_interactions=True)[:, :-1, :-1]  # drop the bias
    var = np.asarray(Xt.columns.map(variable_of))
    names = pd.unique(var)
    to_var = (var[:, None] == names[None, :]).astype(inter.dtype)
    strength = 2 * np.abs(np.einsum("ia,nij,jb->nab", to_var, inter, to_var, optimize=True)).mean(axis=0)
    i, j = np.triu_indices(len(names), k=1)
    return pd.Series(strength[i, j], index=pd.MultiIndex.from_arrays([names[i], names[j]], names=["a", "b"]),
                     name="interaction").sort_values(ascending=False)


def fold_importance(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cfg: dict) -> pd.DataFrame:
    """Mean |SHAP| per input variable on each validation fold, the pipeline refitted on the other folds: how much the
    ranking moves when the training customers change."""
    return pd.DataFrame({
        f"fold {k}": tree_contributions(clone(pipe).fit(X.iloc[tr], y.iloc[tr]), X.iloc[va])[0].abs().mean()
        for k, (tr, va) in enumerate(folds(cfg).split(X, y), start=1)})


def shap_importance(contrib: pd.DataFrame, values: pd.DataFrame, families: dict[str, list[str]],
                    monotonic_rho: float) -> pd.DataFrame:
    """Per input variable: mean |SHAP|, and the Spearman correlation between its value and its SHAP value (numeric
    variables only). '+' / '-' when |rho| reaches monotonic_rho; below it the effect has no single direction."""
    rho = pd.Series({v: values[v].corr(contrib[v], method="spearman") for v in contrib if v in values})
    out = pd.DataFrame({"importance": contrib.abs().mean(), "rho": rho})
    out["direction"] = np.where(out["rho"] >= monotonic_rho, "+", np.where(out["rho"] <= -monotonic_rho, "-", ""))
    out.insert(0, "family", assign_families(list(out.index), families))
    return out.sort_values("importance", ascending=False).rename_axis("variable")
