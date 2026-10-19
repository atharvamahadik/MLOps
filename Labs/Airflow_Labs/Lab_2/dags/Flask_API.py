from __future__ import annotations

import os
import json
import socket
import time
import pendulum
import requests
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from flask import Flask, redirect, render_template

# ---------- Config ----------
# Ensure WEBSERVER points to 'airflow-apiserver' as defined in your compose
WEBSERVER = os.getenv("AIRFLOW_WEBSERVER", "http://airflow-apiserver:8080")
AF_USER   = os.getenv("AIRFLOW_USERNAME", "airflow")
AF_PASS   = os.getenv("AIRFLOW_PASSWORD", "airflow")
TARGET_DAG_ID = os.getenv("TARGET_DAG_ID", "Airflow_Lab2")
FLASK_PORT = 5555
METRICS_PATH = os.getenv("METRICS_PATH", "/opt/airflow/model/metrics.json")

# Set up robust paths for Docker Workers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# ---------- Flask app ----------
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# ---------- Airflow 3 auth (JWT) ----------
_token = {"value": None, "fetched_at": 0.0}
TOKEN_TTL_SECONDS = 15 * 60  # refresh well before expiry


def get_token(force: bool = False) -> str:
    """Fetch (and cache) a JWT from Airflow 3's /auth/token endpoint."""
    now = time.time()
    if force or not _token["value"] or now - _token["fetched_at"] > TOKEN_TTL_SECONDS:
        resp = requests.post(
            f"{WEBSERVER}/auth/token",
            json={"username": AF_USER, "password": AF_PASS},
            timeout=10,
        )
        resp.raise_for_status()
        _token["value"] = resp.json()["access_token"]
        _token["fetched_at"] = now
    return _token["value"]


def load_metrics():
    """Read the metrics.json written by the model DAG. Returns None if unavailable."""
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def get_latest_run_info():
    # Airflow 3 prefers ordering by 'run_after' for the most recent activity    
    url = f"{WEBSERVER}/api/v2/dags/{TARGET_DAG_ID}/dagRuns?order_by=-run_after&limit=1"

    info = {
        "state": "unknown",
        "run_id": "N/A",
        "logical_date": "N/A",
        "start_date": "N/A",
        "end_date": "N/A",
        "note": ""
    }

    try:
        r = None
        for attempt in range(2):  # retry once with a fresh token on 401
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {get_token(force=(attempt == 1))}",
            }
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 401:
                break
        
        if r.status_code != 200:
            info["note"] = f"API Error {r.status_code}"
            return False, info

        data = r.json()
        runs = data.get("dag_runs", [])
        
        if not runs:
            info["note"] = "Waiting for DAG run to appear..."
            return False, info

        run = runs[0]
        state = run.get("state", "unknown")

        print(state)

        info.update({
            "state": state,
            "run_id": run.get("dag_run_id", "N/A"),
            "logical_date": run.get("logical_date") or run.get("run_after", "N/A"),
            "start_date": run.get("start_date", "N/A"),
            "end_date": run.get("end_date", "N/A"),
        })
        
        # LOGIC CHANGE: If it's running, we don't want to say "Failure"
        # We return False to stay on failure/loading page, but update the note
        if state == "queued" or state == "running":
            info["note"] = "Model pipeline is still in progress..."
            return False, info
            
        return state == "success", info

    except Exception as e:
        info["note"] = f"Network issue: {str(e)}"
        return False, info


@app.route("/")
def index():
    ok, _ = get_latest_run_info()
    return redirect("/success" if ok else "/failure")

@app.route("/success")
def success():
    _, info = get_latest_run_info()
    return render_template("success.html", metrics=load_metrics(), **info)

@app.route("/failure")
def failure():
    _, info = get_latest_run_info()
    return render_template("failure.html", **info)

@app.route("/health")
def health():
    return "ok", 200

def port_in_use(port: int) -> bool:
    """True if something is already listening on this port inside the container."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0

def start_flask_app():
    """
    Starts the Flask server. 
    Note: Blocks the worker slot. Disable reloader for Docker.
    If a Flask server is already running on the port, exit cleanly instead of crashing.
    """
    if port_in_use(FLASK_PORT):
        print(f"Port {FLASK_PORT} is already in use; Flask is already running. Nothing to do.", flush=True)
        return
    print(f"Starting Flask on 0.0.0.0:{FLASK_PORT} checking {WEBSERVER}...", flush=True)
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)

# ---------- DAG Definition ----------
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
}

dag = DAG(
    dag_id="Airflow_Lab2_Flask",
    default_args=default_args,
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
)

PythonOperator(
    task_id="start_Flask_API",
    python_callable=start_flask_app,
    dag=dag,
)