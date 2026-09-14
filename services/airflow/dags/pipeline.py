from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

REPO = "/repo"
COMPOSE_FILE = f"{REPO}/code/deployment/docker-compose.yml"

default_args = {
    "owner": "mlops-student",
    # Transient failures (e.g. a network hiccup during docker build)
    # retry automatically without human intervention.
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

# BashOperator is used for all tasks: each runs as a fresh process (clean
# imports) and the script's own logs go straight into the task log.
with DAG(
    dag_id="penguins_pipeline",
    description="Stage 1 data engineering -> Stage 2 model engineering -> Stage 3 deployment",
    schedule="*/5 * * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False, # otherwise Airflow backfills every missed 5-min slot since start_date
    max_active_runs=1, # if a cycle outlasts the schedule, the next run WAITS instead of overlapping
    default_args=default_args,
    tags=["mlops", "assignment"],
    doc_md=__doc__,
) as dag:

    stage1 = BashOperator(
        task_id="stage1_data_engineering",
        bash_command=f"python {REPO}/code/datasets/data_engineering.py",
    )

    stage2 = BashOperator(
    task_id="stage2_model_engineering",
    # MLflow metadata goes to the shared Postgres (SQLite on a Windows
    # bind mount is unreliable). The env var keeps the script itself
    # host-agnostic.
    bash_command=(
        "MLFLOW_TRACKING_URI=postgresql+psycopg2://airflow:airflow@postgres/mlflow "
        f"python {REPO}/code/models/model_engineering.py"
    ),
    # A hung task is killed by the timeout instead of blocking the
    # pipeline forever (max_active_runs=1 would stop all new runs).
    execution_timeout=timedelta(minutes=4),
)

    stage3 = BashOperator(
        task_id="stage3_deployment",
        bash_command=f"docker compose -f {COMPOSE_FILE} up -d --build",
        execution_timeout=timedelta(minutes=4),
    )

    stage1 >> stage2 >> stage3