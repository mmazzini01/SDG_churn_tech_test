"""Weekly churn pipeline: prepare the data once, train each model, evaluate it on the hold-out set, and publish the
churn scores of the production model.

    prepare_data -+-> train_elasticnet -> evaluate_elasticnet
                  +-> train_xgboost    -> evaluate_xgboost -> publish_scores

The logic lives in churn.pipeline; the tasks only pass file paths to each other. The two branches run in parallel.
The split, the metrics and the scores also go to the business database (CHURN_DB_URL in docker-compose.yaml).
"""

from datetime import datetime, timedelta

from airflow.sdk import dag, get_current_context, task


@dag(
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    # Scheduled runs must start without anyone unpausing the DAG in the UI.
    is_paused_upon_creation=False,
    # Every run overwrites the same files on the volume, so two runs must never overlap.
    max_active_runs=1,
    # One retry for transient errors (volume, database); the evaluate tasks override it.
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
)
def churn_pipeline():
    # churn is imported inside each task: this file is re-parsed often, and parsing must not load sklearn and pandas.

    @task(multiple_outputs=True)
    def prepare_data() -> dict[str, str]:
        from churn import config, pipeline

        return pipeline.prepare_data(config.load_config())

    @task
    def train_elasticnet(train_path: str) -> str:
        from churn import config, pipeline

        return pipeline.train_elasticnet(config.load_config(), train_path)

    @task
    def train_xgboost(train_path: str) -> str:
        from churn import config, pipeline

        return pipeline.train_xgboost(config.load_config(), train_path)

    # Metrics go to the task log and the database: XCom carries paths, not results. Each model is scored at its own
    # cut and fails the task if it misses its quality gate. No retry: a missed gate would only fail again.
    @task(retries=0)
    def evaluate_elasticnet(model_path: str, train_path: str, test_path: str) -> None:
        from churn import config, pipeline

        cfg = config.load_config()
        pipeline.evaluate(cfg, model_path, train_path, test_path, cfg["model"]["threshold"],
                          cfg["logreg"]["final"]["min_pr_auc"], get_current_context()["run_id"])

    @task(retries=0)
    def evaluate_xgboost(model_path: str, train_path: str, test_path: str) -> None:
        from churn import config, pipeline

        cfg = config.load_config()
        final = cfg["xgb"]["final"]
        pipeline.evaluate(cfg, model_path, train_path, test_path, final["threshold"], final["min_pr_auc"],
                          get_current_context()["run_id"])

    # Runs only if XGBoost passed its quality gate: a model that got worse never reaches the retention list.
    @task
    def publish_scores(model_path: str, test_path: str) -> None:
        from churn import config, pipeline

        pipeline.publish_scores(config.load_config(), model_path, test_path, get_current_context()["run_id"])

    paths = prepare_data()
    evaluate_elasticnet(train_elasticnet(paths["train"]), paths["train"], paths["test"])
    xgb_model = train_xgboost(paths["train"])
    evaluate_xgboost(xgb_model, paths["train"], paths["test"]) >> publish_scores(xgb_model, paths["test"])


churn_pipeline()
