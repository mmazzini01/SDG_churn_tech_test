"""Engineered features: ratios that take the customer's scale out of a raw signal, the logistic regression's three
domain features, and the XGBoost features (formulas from config)."""

import numpy as np
import pandas as pd


def add_ratios(df: pd.DataFrame, ratios: dict[str, list[str]]) -> pd.DataFrame:
    """One column per ratio, numerator / denominator. A zero denominator gives NaN, not inf: it means no activity,
    which the denominator column itself already shows."""
    out = df.copy()
    for name, (num, den) in ratios.items():
        out[name] = df[num] / df[den].where(df[den] != 0)
    return out


def add_features(X: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The 3 engineered features of the logistic regression. The two flags are yes/no categories, so each gets an
    odds ratio against "no"."""
    lr = cfg["logreg"]
    out = X.copy()
    # Handset older than ~10 months: churn jumps from 37-45% to 60-63% (upgrade cycle). A single coefficient on
    # eqpdays draws a straight line and cannot draw this step; a flag can.
    out["old_handset"] = np.where(X["eqpdays"] >= lr["old_handset_days"], "yes", "no")
    # Months 11-12 of service, the end of the first year (likely a contract renewal): churn 62-65% against 29-40% at
    # 6-10 months. The effect of months rises and falls again, which one coefficient on months cannot follow.
    out["first_year_end"] = np.where(X["months"].isin(lr["first_year_end_months"]), "yes", "no")
    # This month's minutes over the customer's own lifetime average: a customer who calls less than usual is getting
    # ready to leave (4th variable by XGBoost SHAP). The ratio is already defined in config.
    return add_ratios(out, {"mou_vs_lifetime": cfg["features"]["ratios"]["mou_vs_lifetime"]})


def add_features_xgb(X: pd.DataFrame, formulas: dict[str, str]) -> pd.DataFrame:
    """One column per formula, a pandas expression on the raw columns (xgb in config). A division by zero gives inf
    (NaN for 0 / 0); both become NaN, which XGBoost sends down the branch it learned for missing values."""
    out = X.copy()
    for name, formula in formulas.items():
        out[name] = X.eval(formula).replace([np.inf, -np.inf], np.nan)
    return out
