import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from churn import model
from churn.features import add_ratios

CFG = {
    "target": "churn",
    "exclude": ["id"],
    "seed": 0,
    "cleaning": {"unknown_codes": {"cat": ["U"]}, "clip_lower": {"x": 0}},
    "features": {"ratios": {"x_per_z": ["x", "z"]}},
    "model": {"cv_folds": 2, "threshold": 0.5, "missing_indicators": ["x"], "min_frequency": 0.05,
              "baseline": {"C": 1.0, "max_iter": 500}},
}


def _data(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(5, 2, n)
    df = pd.DataFrame({
        "id": np.arange(n),
        "x": np.where(rng.random(n) < 0.1, np.nan, x),
        "z": rng.integers(0, 3, n).astype(float),
        "cat": rng.choice(["A", "B", "U", None], n),
        "churn": (x + rng.normal(0, 2, n) > 5).astype(int),
    })
    return df


def test_add_ratios_zero_denominator_is_nan_not_inf():
    df = pd.DataFrame({"a": [2.0, 1.0, 3.0], "b": [4.0, 0.0, np.nan]})
    out = add_ratios(df, {"a_per_b": ["a", "b"]})
    assert out["a_per_b"].iloc[0] == 0.5 and out["a_per_b"].iloc[1:].isna().all()
    assert "a_per_b" not in df  # input left untouched


def test_pipeline_scores_unseen_levels_and_contributions_add_up():
    train, new = _data(), _data(20)
    new.loc[0, "cat"] = "never_seen"
    X, y = model.split_xy(train, CFG)
    pipe = model.build_pipeline(CFG, model.baseline(CFG)).fit(X, y)
    X_new, _ = model.split_xy(new, CFG)
    proba = pipe.predict_proba(X_new)[:, 1]
    assert np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all()
    contrib = model.contributions(pipe, X_new)
    # Every encoded column lands in exactly one variable: contributions plus intercept rebuild the log-odds.
    assert {"x", "x (missing)", "x_per_z", "cat"} <= set(contrib.columns) and "id" not in contrib.columns
    np.testing.assert_allclose(contrib.sum(axis=1) + pipe[-1].intercept_[0], pipe.decision_function(X_new))


def test_gam_splines_map_back_to_their_variable_and_contributions_add_up():
    cfg = {**CFG, "model": {**CFG["model"], "gam": {"degree": 3}}}
    train, new = _data(), _data(20)
    new.loc[1, "x"] = 1e6  # far past the training range: the curve stays flat instead of extrapolating a cubic
    X, y = model.split_xy(train, cfg)
    pipe = model.build_pipeline(cfg, model.baseline(cfg), n_knots=4).fit(X, y)
    assert any(c.startswith("x_sp_") for c in pipe["scale"].get_feature_names_out())
    X_new, _ = model.split_xy(new, cfg)
    contrib = model.contributions(pipe, X_new)
    assert set(contrib.columns) == {"x", "x (missing)", "z", "x_per_z", "cat"}
    assert np.isfinite(contrib.to_numpy()).all()
    np.testing.assert_allclose(contrib.sum(axis=1) + pipe[-1].intercept_[0], pipe.decision_function(X_new))


def test_tensor_term_is_one_variable_and_joint_effect_ignores_the_other_columns():
    cfg = {**CFG, "model": {**CFG["model"], "gam": {"degree": 3, "tensor_knots": 4}}}
    X, y = model.split_xy(_data(), cfg)
    pipe = model.build_pipeline(cfg, model.baseline(cfg), n_knots=4, tensors=[["x", "z"]]).fit(X, y)
    contrib = model.contributions(pipe, X)
    # 5 basis columns per variable (4 knots, cubic, no bias) -> 25 products, all in one term.
    assert pd.Index(pipe["scale"].get_feature_names_out()).str.startswith("x × z_tp_").sum() == 25
    assert "x × z" in contrib.columns
    np.testing.assert_allclose(contrib.sum(axis=1) + pipe[-1].intercept_[0], pipe.decision_function(X))
    # Additive model: the pair's joint effect is the same whichever customer fills the other columns.
    one, two = (model.joint_effect(pipe, X.iloc[[i]], "x", [3.0, 5.0, 7.0], "z", [0.0, 2.0]) for i in (0, 1))
    assert one.shape == (2, 3)
    pd.testing.assert_frame_equal(one, two)
    # x = 100 is far outside the customers' values on either line: blanked; the observed values are kept.
    kept = model.on_support(model.joint_effect(pipe, X.iloc[[0]], "x", [5.0, 100.0], "z", [0.0, 2.0]), X, n=50)
    assert kept[100.0].isna().all() and kept[5.0].notna().all()


def test_pick_one_se_takes_the_sparsest_within_one_se():
    path = pd.DataFrame({
        "l1_ratio": [0.5, 0.5, 0.5, 0.8],
        "C": [0.01, 0.1, 1.0, 0.1],
        "score": [-0.70, -0.665, -0.660, -0.662],
        "se": [0.002, 0.002, 0.002, 0.002],
        "n_nonzero": [5, 40, 90, 30],
    })
    chosen = model.pick_one_se(path)
    # Within 1 SE of the best (-0.660): -0.662 (30 columns) and -0.660 (90); -0.665 is outside.
    assert (chosen["C"], chosen["l1_ratio"], chosen["n_nonzero"]) == (0.1, 0.8, 30)
    assert chosen["score"] == pytest.approx(-0.662)


def test_tree_pipeline_keeps_nan_scores_unseen_levels_and_shap_adds_up():
    cfg = {**CFG, "model": {**CFG["model"], "xgboost": {"params": {"learning_rate": 0.3, "n_jobs": 1}}}}
    train, new = _data(), _data(20)
    new.loc[0, "cat"] = "never_seen"
    X, y = model.split_xy(train, cfg)
    pipe = model.build_tree_pipeline(cfg, model.xgboost(cfg, 3, 20)).fit(X, y)
    X_new, _ = model.split_xy(new, cfg)
    # NaN reaches the trees untouched: no imputation in this pipeline.
    assert pipe[:-1].transform(X_new)["x"].isna().any()
    contrib, base = model.tree_contributions(pipe, X_new)
    assert set(contrib.columns) == {"x", "z", "x_per_z", "cat"}
    # Base value plus one row of contributions rebuilds that customer's log-odds (SHAP additivity).
    np.testing.assert_allclose(base + contrib.sum(axis=1), logit(pipe.predict_proba(X_new)[:, 1]), atol=1e-4)


def test_best_f1_threshold_picks_the_cut_with_the_highest_f1():
    y = pd.Series([0, 0, 1, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.9])
    # Cut 0.3: precision 3/4, recall 1, F1 0.857; cut 0.6: 1 and 2/3, F1 0.8; cut 0.1: 1/2 and 1, F1 0.667.
    assert model.best_f1_threshold(y, proba) == pytest.approx(0.3)


def test_tree_interactions_finds_the_pair_the_target_depends_on_jointly():
    cfg = {**CFG, "model": {**CFG["model"], "xgboost": {"params": {"learning_rate": 0.3, "n_jobs": 1}}}}
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"id": np.arange(600), "x": rng.normal(size=600), "z": rng.normal(size=600),
                       "cat": rng.choice(["A", "B"], 600)})
    df["churn"] = ((df["x"] > 0) ^ (df["z"] > 0)).astype(int)  # pure interaction: neither variable matters alone
    X, y = model.split_xy(df, cfg)
    pipe = model.build_tree_pipeline(cfg, model.xgboost(cfg, 3, 30), ratios=False).fit(X, y)
    pairs = model.tree_interactions(pipe, X)
    assert pairs.index[0] in {("x", "z"), ("z", "x")}
