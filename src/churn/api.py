"""Churn simulation API: how a hold-out customer's churn probability moves when one variable changes. It scores with
the XGBoost pipeline the DAG pickled, so the probability before the change is the one in the churn_scores table.

Served by the churn-api container (docker-compose.yaml): Swagger on http://localhost:8000/docs.
"""

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from churn.config import ROOT, load_config
from churn.model import split_xy
from churn.pipeline import _load

app = FastAPI(
    title="Churn simulation API",
    description="Change one variable of a customer and see how the model's churn probability moves. It is the "
                "model's response, not a causal effect: what an offer really changes is measured with an A/B test.",
)


class SimulationRequest(BaseModel):
    variable: str = Field(description="A model input, named as in the dataset", examples=["eqpdays"])
    value: float | str = Field(description="The new value: a number, or a level of a categorical variable",
                               examples=[0])


class SimulationResponse(BaseModel):
    customer_id: int
    variable: str
    old_value: float | str | None
    new_value: float | str
    churn_probability_before: float
    churn_probability_after: float
    change: float = Field(description="after - before (0.10 = 10 percentage points)")
    flagged_before: bool = Field(description="On the retention list at the production cut")
    flagged_after: bool


def simulate(pipe: Pipeline, row: pd.DataFrame, variable: str, value: float | str) -> tuple[float, float]:
    """Churn probability of one customer (a one-row frame of raw inputs) before and after setting variable to value.
    The whole pipeline runs again, so the engineered ratios follow the changed variable."""
    changed = row.assign(**{variable: value})
    before, after = pipe.predict_proba(pd.concat([row, changed]))[:, 1]
    return float(before), float(after)


def _parse(X: pd.DataFrame, variable: str, value: float | str) -> float | str:
    """Check the new value against the model inputs: a wrong type or an unseen level would be scored silently."""
    if variable not in X.columns:
        raise HTTPException(422, f"{variable} is not a model input (Customer_ID and ethnic never are)")
    if pd.api.types.is_numeric_dtype(X[variable]):
        if isinstance(value, str):
            raise HTTPException(422, f"{variable} is numeric")
        return value
    levels = set(X[variable].dropna())
    if value not in levels:
        raise HTTPException(422, f"{variable} takes one of {sorted(levels)}")
    return value


@app.post("/customers/{customer_id}/simulate")
def simulate_customer(customer_id: int, req: SimulationRequest) -> SimulationResponse:
    """customer_id: a hold-out customer (the churn_scores table)."""
    cfg = load_config()
    model_path, test_path = ROOT / cfg["paths"]["xgb_model"], ROOT / cfg["paths"]["test"]
    if not (model_path.exists() and test_path.exists()):
        raise HTTPException(503, "no trained model yet: run the churn_pipeline DAG")
    # ponytail: model and hold-out read on every request (~0.3 s), so a weekly retrain is served with no restart;
    # cache them by file modification time if latency matters.
    pipe, test = _load(str(model_path)), pd.read_pickle(test_path)
    X, _ = split_xy(test, cfg)
    row = X[test[cfg["id"]] == customer_id]
    if row.empty:
        raise HTTPException(404, f"customer {customer_id} is not in the hold-out set")
    value = _parse(X, req.variable, req.value)
    before, after = simulate(pipe, row, req.variable, value)
    cut = cfg["xgb"]["final"]["threshold"]
    return SimulationResponse(
        customer_id=customer_id, variable=req.variable, old_value=row[req.variable].item(), new_value=value,
        churn_probability_before=before, churn_probability_after=after, change=after - before,
        flagged_before=before >= cut, flagged_after=after >= cut,
    )
