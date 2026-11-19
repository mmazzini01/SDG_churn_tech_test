"""Weekly churn pipeline: prepare the data, train an XGBoost candidate, evaluate it on the hold-out set, promote it to
production if it passes the quality gate, and publish the churn scores.

    prepare_data -> train_xgboost -> evaluate_xgboost -> promote_xgboost -> publish_scores

The logic lives in churn.pipeline; the tasks only pass file paths to each other. The split, the metrics and the scores
also go to the business database (CHURN_DB_URL in docker-compose.yaml).
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
    # One retry for transient errors (volume, database); evaluate overrides it.
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
)
def churn_pipeline():
    # churn is imported inside each task: this file is re-parsed often, and parsing must not load sklearn and pandas.

    @task(multiple_outputs=True)
    def prepare_data() -> dict[str, str]:
        from churn import config, pipeline

        return pipeline.prepare_data(config.load_config())

    @task
    def train_xgboost(train_path: str) -> str:
        from churn import config, pipeline

        return pipeline.train_xgboost(config.load_config(), train_path)

    # Metrics go to the task log and the database: XCom carries paths, not results. The task fails if the candidate
    # misses its quality gate. No retry: a missed gate would only fail again.
    @task(retries=0)
    def evaluate_xgboost(candidate_path: str, train_path: str, test_path: str) -> None:
        from churn import config, pipeline

        cfg = config.load_config()
        final = cfg["xgb"]["final"]
        pipeline.evaluate(cfg, candidate_path, train_path, test_path, final["threshold"], final["min_pr_auc"],
                          get_current_context()["run_id"])

    # Runs only if the candidate passed the gate: otherwise the API and the retention list keep the last approved model.
    @task
    def promote_xgboost(candidate_path: str) -> str:
        from churn import config, pipeline

        return pipeline.promote(config.load_config(), candidate_path)

    @task
    def publish_scores(model_path: str, test_path: str) -> None:
        from churn import config, pipeline

        pipeline.publish_scores(config.load_config(), model_path, test_path, get_current_context()["run_id"])

    paths = prepare_data()
    candidate = train_xgboost(paths["train"])
    model = promote_xgboost(candidate)
    evaluate_xgboost(candidate, paths["train"], paths["test"]) >> model
    publish_scores(model, paths["test"])


churn_pipeline()
