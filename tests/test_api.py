import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from churn import api


def test_simulate_rescores_the_changed_customer():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=200), "cat": rng.choice(["A", "B"], 200)})
    pipe = make_pipeline(ColumnTransformer([("x", "passthrough", ["x"])]), LogisticRegression())
    pipe.fit(X, (X["x"] > 0).astype(int))
    row = X.iloc[[0]]

    before, after = api.simulate(pipe, row, "x", 3.0)
    assert before == pytest.approx(pipe.predict_proba(row)[0, 1])  # before = the customer's current score
    assert after > 0.9 and row["x"].item() != 3.0                   # x drives churn; the input row is unchanged

    assert api._parse(X, "cat", "B") == "B"
    for variable, value in [("ethnic", 1.0), ("x", "high"), ("cat", "Z")]:
        with pytest.raises(HTTPException) as err:
            api._parse(X, variable, value)
        assert err.value.status_code == 422
