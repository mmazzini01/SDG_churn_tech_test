set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# ============================================================================================================
# Presentation: the live demo, one recipe after the other (`just --list --unsorted` shows them in this order).
# Before the talk: start Docker Desktop and run `just up` once; the DAG runs by itself at the first start.
# ============================================================================================================

# 1. Build the image and start Airflow on http://localhost:8080 (airflow / airflow)
[group('presentation')]
up:
    docker compose up -d --build

# 2. Open the Airflow UI
[group('presentation')]
airflow-ui:
    uv run python -m webbrowser http://localhost:8080

# 3. Run the DAG 
[group('presentation')]
trigger:
    docker compose exec airflow-scheduler airflow dags trigger churn_pipeline -o yaml

# Queries to paste at the psql prompt (\q to quit):
#   SELECT model, round(pr_auc::numeric, 4) AS pr_auc, passed, evaluated_at FROM model_metrics ORDER BY evaluated_at DESC LIMIT 4;
#   SELECT suggested_action, count(*) FROM churn_scores WHERE flagged GROUP BY 1 ORDER BY 2 DESC;
# 4. Open psql on the business database (split, model_metrics, churn_scores)
[group('presentation')]
db:
    docker compose exec churn-db psql -U churn -d churn


# 5. Open the MLflow UI on http://localhost:5000
[group('presentation')]
mlflow-ui:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# In Swagger: POST /customers/{customer_id}/simulate -> "Try it out", then paste
#   customer_id  1012140
#   body         {"variable": "eqpdays", "value": 0}     new handset: 0.55 -> 0.36, drops off the retention list
#   body         {"variable": "mou_Mean", "value": 0}    no usage this month: 0.55 -> 0.71
# 6. Open the churn simulation API (Swagger) on http://localhost:8000/docs; needs one finished DAG run
[group('presentation')]
swagger:
    uv run python -m webbrowser http://localhost:8000/docs

# 7. Stop the stack; the metadata DB volume is kept
[group('presentation')]
down:
    docker compose down

# ============================================================================================================
# Development: tests, notebooks and the offline experiments.
# ============================================================================================================

# Run the unit tests
[group('development')]
test:
    uv run pytest

# Lint the code and the notebooks
[group('development')]
lint:
    uvx ruff check .

# Re-run the EDA notebook end to end, saving its outputs in place
[group('development')]
eda:
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

# Tune (Optuna, 2 x 20 trials), compare and train the logistic regression, tracked in MLflow (about 15 minutes)
[group('development')]
logistic:
    uv run python -m churn.train_logistic

# Re-run the logistic interpretation notebook (loads the final model from MLflow), saving its outputs in place
[group('development')]
logistic-notebook:
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_logistic_interpretation.ipynb

# Compare the candidate engineered features for XGBoost, training set only (about 10 minutes)
[group('development')]
xgb-screen:
    uv run python -m churn.train_xgboost --screen

# Tune (Optuna, 2 x 40 trials), compare and train XGBoost, tracked in MLflow (about 30 minutes)
[group('development')]
xgb:
    uv run python -m churn.train_xgboost

# Re-run the XGBoost interpretation notebook (loads the final model from MLflow), saving its outputs in place
[group('development')]
xgb-notebook:
    uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/03_xgboost_interpretation.ipynb
