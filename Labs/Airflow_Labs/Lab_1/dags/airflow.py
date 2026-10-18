from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

from src.lab import (
    load_data,
    data_preprocessing,
    compute_sse,
    find_optimal_k,
    train_final_model,
    predict_test,
)

default_args = {
    "owner": "Atharva Mahadik",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="Airflow_Lab1",
    description="K-Means customer segmentation with elbow-based k selection",
    doc_md=__doc__,
    default_args=default_args,
    start_date=datetime(2026, 1, 15),
    schedule=None,
    catchup=False,
    tags=["mlops", "lab1", "kmeans"],
) as dag:

    load = PythonOperator(task_id="load_data", python_callable=load_data)

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=data_preprocessing,
        op_args=[load.output],
    )

    sse = PythonOperator(
        task_id="compute_sse",
        python_callable=compute_sse,
        op_args=[preprocess.output],
    )

    best_k = PythonOperator(
        task_id="find_optimal_k",
        python_callable=find_optimal_k,
        op_args=[sse.output],
    )

    train = PythonOperator(
        task_id="train_final_model",
        python_callable=train_final_model,
        op_args=[preprocess.output, best_k.output],
    )

    predict = PythonOperator(
        task_id="predict_test",
        python_callable=predict_test,
        op_args=[train.output],
    )

    load >> preprocess >> sse >> best_k >> train >> predict


if __name__ == "__main__":
    dag.test()