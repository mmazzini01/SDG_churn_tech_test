"""What the training scripts, the notebooks and the DAG share: the features / target split, the CV folds, the metrics
of CV and of the hold-out set, and XGBoost's per-variable TreeSHAP explanations."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import DMatrix


def split_xy(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Features and target; the key and the excluded columns never reach the model."""
    return df.drop(columns=[cfg["target"], *cfg["exclude"]]), df[cfg["target"]]


def _level_name(feature: str, category: str) -> str:
    # "variable=level" lets every one-hot column be traced back to its variable (see variable_of). The pickled XGBoost
    # pipeline refers to this function by its module path, so it stays here.
    return f"{feature}={category}"


def folds(cfg: dict) -> StratifiedKFold:
    """The one set of folds used for tuning and for comparing models."""
    return StratifiedKFold(cfg["model"]["cv_folds"], shuffle=True, random_state=cfg["seed"])


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


def oof_proba(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cfg: dict) -> np.ndarray:
    """Out-of-fold probabilities: each training customer scored by the pipeline fitted on the other folds."""
    return cross_val_predict(pipe, X, y, cv=folds(cfg), method="predict_proba", n_jobs=cfg["model"]["cv_folds"])[:, 1]


def variable_of(column: str) -> str:
    """Input variable of an encoded column: 'crclscod=A' -> 'crclscod'; a numeric column is its own variable."""
    return column.split("=")[0]


def tree_contributions(pipe: Pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Exact TreeSHAP values (log-odds) from XGBoost's own implementation, one-hot columns summed per input variable,
    and the base value they start from: base + a customer's contributions = the model's log-odds for that customer."""
    Xt = pipe[:-1].transform(X)
    shap = pipe[-1].get_booster().predict(DMatrix(Xt), pred_contribs=True)
    # The last column is the bias: the same expected log-odds for every customer.
    contrib = pd.DataFrame(shap[:, :-1], index=X.index, columns=Xt.columns).T.groupby(variable_of).sum().T
    return contrib, float(shap[0, -1])


def shap_importance(contrib: pd.DataFrame, values: pd.DataFrame, monotonic_rho: float) -> pd.DataFrame:
    """Per input variable: mean |SHAP|, and the Spearman correlation between its value and its SHAP value (numeric
    variables only). '+' / '-' when |rho| reaches monotonic_rho; below it the effect has no single direction."""
    rho = pd.Series({v: values[v].corr(contrib[v], method="spearman") for v in contrib if v in values})
    out = pd.DataFrame({"importance": contrib.abs().mean(), "rho": rho})
    out["direction"] = np.where(out["rho"] >= monotonic_rho, "+", np.where(out["rho"] <= -monotonic_rho, "-", ""))
    return out.sort_values("importance", ascending=False).rename_axis("variable")
