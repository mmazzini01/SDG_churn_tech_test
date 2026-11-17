set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Build the image and start Airflow on http://localhost:8080 (airflow / airflow)
up:
    docker compose up -d --build

# Stop the stack; the metadata DB volume is kept
down:
    docker compose down

# Run the DAG now, without waiting for its weekly schedule
trigger:
    docker compose exec airflow-scheduler airflow dags trigger churn_pipeline

# Open psql on the business database (split, model_metrics, churn_scores)
db:
    docker compose exec churn-db psql -U churn -d churn

# Run the unit tests
test:
    uv run pytest

# Lint the code and the notebooks
lint:
    uvx ruff check .

# Re-run the EDA notebook end to end, saving its outputs in place
eda:
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

# Re-run the modelling notebook (logistic regression + elastic net, about 10 minutes), saving its outputs in place
model:
    uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/02_modeling.ipynb

# Re-run the XGBoost notebook (Bayesian tuning + SHAP, about an hour), saving its outputs in place
xgboost:
    uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=5400 notebooks/03_xgboost.ipynb

# Tune (Optuna, 2 x 20 trials), compare and train the logistic regression, tracked in MLflow (about 15 minutes)
logistic:
    uv run python -m churn.train_logistic

# Re-run the logistic interpretation notebook (loads the final model from MLflow), saving its outputs in place
logistic-notebook:
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_logistic_interpretation.ipynb

# Compare the candidate engineered features for XGBoost, training set only (about 10 minutes)
xgb-screen:
    uv run python -m churn.train_xgboost --screen

# Tune (Optuna, 2 x 40 trials), compare and train XGBoost, tracked in MLflow (about 30 minutes)
xgb:
    uv run python -m churn.train_xgboost

# Re-run the XGBoost interpretation notebook (loads the final model from MLflow), saving its outputs in place
xgb-notebook:
    uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/06_xgboost_interpretation.ipynb

# Open the MLflow UI on http://localhost:5000
mlflow-ui:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Re-run the GAM notebook (splines + logistic regression, about 12 minutes), saving its outputs in place
gam:
    uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/04_gam.ipynb
