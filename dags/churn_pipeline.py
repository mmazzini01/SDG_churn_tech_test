"""Weekly churn pipeline: prepare data, train the model, evaluate it on the hold-out set."""

import logging
from datetime import datetime
from pathlib import Path

from airflow.sdk import dag, task

log = logging.getLogger(__name__)

# ponytail: placeholder tasks that only prove the mounts work; replace with churn.* calls in the integration phase
DATASET = Path("/opt/airflow/data/raw/dataset.csv")


@dag(
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    # Scheduled runs must start without anyone unpausing the DAG in the UI.
    is_paused_upon_creation=False,
)
def churn_pipeline():
    @task
    def prepare_data() -> None:
        if not DATASET.exists():
            raise FileNotFoundError(f"Dataset not mounted: {DATASET}")
        log.info("Dataset found: %s (%.1f MB)", DATASET, DATASET.stat().st_size / 1e6)

    @task
    def train_model() -> None:
        # Imported inside the task to keep DAG parsing light.
        import churn

        log.info("churn package imported from %s", churn.__file__)

    @task
    def evaluate_model() -> None:
        log.info("Evaluation placeholder: hold-out metrics will be logged here")

    prepare_data() >> train_model() >> evaluate_model()


churn_pipeline()
