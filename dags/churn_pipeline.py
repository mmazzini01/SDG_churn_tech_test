"""Weekly churn pipeline: prepare the data once, then train each model and evaluate it on the hold-out set.

    prepare_data -+-> train_elasticnet -> evaluate_elasticnet
                  +-> train_xgboost    -> evaluate_xgboost

The logic lives in churn.pipeline; the tasks only pass file paths to each other. The two branches run in parallel.
"""

from datetime import datetime

from airflow.sdk import dag, task


@dag(
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    # Scheduled runs must start without anyone unpausing the DAG in the UI.
    is_paused_upon_creation=False,
    # Every run overwrites the same files on the volume, so two runs must never overlap.
    max_active_runs=1,
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

    # Metrics go to the task log only: XCom carries paths, not results. Each model is scored at its own cut.
    @task
    def evaluate_elasticnet(model_path: str, train_path: str, test_path: str) -> None:
        from churn import config, pipeline

        cfg = config.load_config()
        pipeline.evaluate(cfg, model_path, train_path, test_path, cfg["model"]["threshold"])

    @task
    def evaluate_xgboost(model_path: str, train_path: str, test_path: str) -> None:
        from churn import config, pipeline

        cfg = config.load_config()
        pipeline.evaluate(cfg, model_path, train_path, test_path, cfg["xgb"]["final"]["threshold"])

    paths = prepare_data()
    evaluate_elasticnet(train_elasticnet(paths["train"]), paths["train"], paths["test"])
    evaluate_xgboost(train_xgboost(paths["train"]), paths["train"], paths["test"])


churn_pipeline()
