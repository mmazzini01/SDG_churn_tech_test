import pandas as pd

from churn.features import add_features
from churn.train_logistic import keep_features

CFG = {
    "logreg": {"old_handset_days": 300, "first_year_end_months": [11, 12]},
    "features": {"ratios": {"mou_vs_lifetime": ["mou_Mean", "avgmou"]}},
}


def test_add_features():
    X = pd.DataFrame({"eqpdays": [299, 300], "months": [10, 12], "mou_Mean": [50.0, 10.0], "avgmou": [100.0, 0.0]})
    out = add_features(X, CFG)
    assert out["old_handset"].tolist() == ["no", "yes"]
    assert out["first_year_end"].tolist() == ["no", "yes"]
    assert out["mou_vs_lifetime"].iloc[0] == 0.5
    assert pd.isna(out["mou_vs_lifetime"].iloc[1])  # no lifetime usage: NaN, not inf


def test_keep_features_needs_more_than_one_sd():
    base = {"cv_pr_auc_mean": 0.620, "cv_pr_auc_std": 0.004}
    assert not keep_features(base, {"cv_pr_auc_mean": 0.623})
    assert keep_features(base, {"cv_pr_auc_mean": 0.625})
