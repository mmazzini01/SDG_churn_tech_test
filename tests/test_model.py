import numpy as np
import pandas as pd
from scipy.special import logit

from churn import model
from churn.features import add_ratios
from churn.pipeline import build_xgb_pipeline

CFG = {
    "target": "churn",
    "exclude": ["id"],
    "seed": 0,
    "cleaning": {"unknown_codes": {"cat": ["U"]}, "clip_lower": {"x": 0}},
    "xgb": {"params": {"learning_rate": 0.3, "n_jobs": 1}},
}


def _data(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(5, 2, n)
    return pd.DataFrame({
        "id": np.arange(n),
        "x": np.where(rng.random(n) < 0.1, np.nan, x),
        "z": rng.integers(0, 3, n).astype(float),
        "cat": rng.choice(["A", "B", "U", None], n),
        "churn": (x + rng.normal(0, 2, n) > 5).astype(int),
    })


def test_add_ratios_zero_denominator_is_nan_not_inf():
    df = pd.DataFrame({"a": [2.0, 1.0, 3.0], "b": [4.0, 0.0, np.nan]})
    out = add_ratios(df, {"a_per_b": ["a", "b"]})
    assert out["a_per_b"].iloc[0] == 0.5 and out["a_per_b"].iloc[1:].isna().all()
    assert "a_per_b" not in df  # input left untouched


def test_xgb_pipeline_keeps_nan_scores_unseen_levels_and_shap_adds_up():
    train, new = _data(), _data(20)
    new.loc[0, "cat"] = "never_seen"
    X, y = model.split_xy(train, CFG)
    pipe = build_xgb_pipeline(CFG, {"n_estimators": 20, "max_depth": 3}, {"x_per_z": "x / z"}).fit(X, y)
    X_new, _ = model.split_xy(new, CFG)
    # NaN reaches the trees untouched: no imputation in this pipeline.
    assert pipe[:-1].transform(X_new)["x"].isna().any()
    contrib, base = model.tree_contributions(pipe, X_new)
    # The one-hot columns ("cat=A", ...) are summed back into their variable; the key never reaches the model.
    assert set(contrib.columns) == {"x", "z", "x_per_z", "cat"}
    # Base value plus one row of contributions rebuilds that customer's log-odds (SHAP additivity).
    np.testing.assert_allclose(base + contrib.sum(axis=1), logit(pipe.predict_proba(X_new)[:, 1]), atol=1e-4)
