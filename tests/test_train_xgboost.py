import numpy as np
import pandas as pd

from churn.features import add_features_xgb
from churn.train_xgboost import recall_threshold


def test_add_features_xgb_division_by_zero_is_nan():
    X = pd.DataFrame({"mou_Mean": [50.0, 10.0, 0.0], "avgmou": [100.0, 0.0, 0.0],
                      "drop_vce_Mean": [1.0, 2.0, 0.0], "plcd_vce_Mean": [3.0, 0.0, 0.0], "recv_vce_Mean": [1.0, 0.0, 0.0]})
    out = add_features_xgb(X, {"ratio": "mou_Mean / avgmou", "rate": "drop_vce_Mean / (plcd_vce_Mean + recv_vce_Mean)"})
    assert out["ratio"].iloc[0] == 0.5
    assert out["rate"].iloc[0] == 0.25
    assert out[["ratio", "rate"]].iloc[1:].isna().all().all()  # x / 0 and 0 / 0: NaN, not inf
    assert list(X.columns) == ["mou_Mean", "avgmou", "drop_vce_Mean", "plcd_vce_Mean", "recv_vce_Mean"]  # input kept


def test_recall_threshold_lowest_cut_with_enough_precision():
    y = pd.Series([0, 1, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert recall_threshold(y, proba, 0.6) == 0.1   # everyone flagged: precision 3/5, recall 1
    assert recall_threshold(y, proba, 0.7) == 0.2   # precision 3/4, recall 1
    assert recall_threshold(y, proba, 1.0) == 0.8   # precision 1, recall 2/3
