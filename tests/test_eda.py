import numpy as np
import pandas as pd
import pytest

from churn import eda


def test_assign_families_longest_prefix_wins():
    families = {"usage": ["avg"], "revenue": ["avgrev"]}
    assert eda.assign_families(["avgrev", "avgmou"], families).to_dict() == {"avgrev": "revenue", "avgmou": "usage"}
    with pytest.raises(ValueError):
        eda.assign_families(["unknown_col"], families)


def test_churn_by_bin_rates_odds_ratio_and_missing_bin():
    df = pd.DataFrame({
        "x": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, np.nan, np.nan],
        "churn": [1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    })
    out = eda.churn_by_bin(df, "x", "churn", bins=10).set_index("bin")
    assert out["churn_rate"].to_dict() == {"1": 0.75, "2": 0.25, "Missing": 0.5}
    assert out["n"].sum() == len(df)
    # Overall odds are 1, so each bin's odds ratio is its own odds: 0.75/0.25 = 3 and 0.25/0.75 = 1/3.
    assert out["odds_ratio"].to_dict() == pytest.approx({"1": 3.0, "2": 1 / 3, "Missing": 1.0})


def test_rank_vs_target_auc_direction_and_categorical_with_nan():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 400))
    df = pd.DataFrame({
        "churn": y,
        "perfect": y.astype(float),
        "inverse": -y.astype(float),
        "cat": pd.Series(np.where(y == 1, "a", "b")).where(rng.random(400) > 0.1),
    })
    families = pd.Series({"perfect": "f", "inverse": "f", "cat": "f"})
    rank = eda.rank_vs_target(df, "churn", families, n_bins=10, seed=0).set_index("variable")
    assert rank.loc["perfect", "auc"] == 1.0 and rank.loc["perfect", "direction"] == "+"
    assert rank.loc["inverse", "auc"] == 0.0 and rank.loc["inverse", "direction"] == "-"
    assert rank.loc["cat", "auc_strength"] > 0.4


def test_tests_vs_target_effect_sizes_and_rare_levels():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 400))
    df = pd.DataFrame({
        "churn": y,
        "x": y + rng.normal(0, 1, 400),
        # One customer in level "z": expected count < 5, so it is pooled for the chi-square test.
        "cat": np.where(y == 1, "a", "b"),
    })
    df.loc[0, "cat"] = "z"
    out = eda.tests_vs_target(df, "churn", min_expected=5).set_index("variable")
    rank = eda.rank_vs_target(df, "churn", pd.Series({"x": "f", "cat": "f"}), n_bins=10, seed=0).set_index("variable")
    assert out.loc["x", "effect_size"] == pytest.approx(2 * rank.loc["x", "auc"] - 1)
    assert out.loc["cat", "levels_merged"] == 1 and out.loc["cat", "effect_size"] > 0.9
    assert (out["p_fdr"] >= out["p_value"]).all()


def test_redundancy_groups_and_greedy_candidates():
    # a~b and b~c are strong, a~c is not: one chained group whose weakest pair is 0.5.
    corr = pd.DataFrame([[1, 0.95, 0.5, 0], [0.95, 1, 0.95, 0], [0.5, 0.95, 1, 0], [0, 0, 0, 1]],
                        index=list("abcd"), columns=list("abcd"), dtype=float)
    families = pd.Series({"a": "f", "b": "f", "c": "g", "d": "g"})
    groups = eda.redundancy_groups(corr, 0.8, families)
    assert groups.to_dict("records") == [{"n_variables": 3, "min_abs_corr": 0.5, "families": "f, g", "variables": "a, b, c"}]
    # Greedy by MI: a kept, b dropped as its twin, c kept (only 0.5 with a), d negligible.
    rank = pd.DataFrame({"variable": list("abcd"), "family": "f", "mi": [0.4, 0.3, 0.2, 0.001],
                         "auc_strength": [0.1, 0.1, 0.1, 0.001], "rank_mi": [1, 2, 3, 4]})
    out = eda.removal_candidates(rank, corr, extreme=0.9, negligible=0.01).set_index("variable")["reason"]
    assert set(out.index) == {"b", "d"}
    assert out["b"].startswith("near-duplicate of a") and out["d"].startswith("negligible")


def test_bin_labels_rounded_to_two_decimals():
    assert eda._label(pd.Interval(-1.4869999999999999, -0.573)) == "(-1.49, -0.57]"
    assert eda._label(pd.Interval(-0.001, 93.0)) == "(0, 93]"  # no "-0", no trailing zeros
    assert eda._label(12.0) == "12" and eda._label("A") == "A"
