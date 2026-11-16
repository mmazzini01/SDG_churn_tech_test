"""Engineered features: ratios that take the customer's scale out of a raw signal."""

import pandas as pd


def add_ratios(df: pd.DataFrame, ratios: dict[str, list[str]]) -> pd.DataFrame:
    """One column per ratio, numerator / denominator. A zero denominator gives NaN, not inf: it means no activity,
    which the denominator column itself already shows."""
    out = df.copy()
    for name, (num, den) in ratios.items():
        out[name] = df[num] / df[den].where(df[den] != 0)
    return out
