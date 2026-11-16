set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Build the image and start Airflow on http://localhost:8080 (airflow / airflow)
up:
    docker compose up -d --build

# Stop the stack; the metadata DB volume is kept
down:
    docker compose down

# Run the unit tests
test:
    uv run pytest

# Re-run the EDA notebook end to end, saving its outputs in place
eda:
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

# Re-run the modelling notebook (logistic regression + elastic net, about 10 minutes), saving its outputs in place
model:
    uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/02_modeling.ipynb

# Re-run the XGBoost notebook (Bayesian tuning + SHAP, about an hour), saving its outputs in place
xgboost:
    uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=5400 notebooks/03_xgboost.ipynb

# Re-run the GAM notebook (splines + logistic regression, about 12 minutes), saving its outputs in place
gam:
    uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/04_gam.ipynb
