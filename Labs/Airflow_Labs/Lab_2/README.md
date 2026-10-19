# My Changes

This is my version of Airflow Lab 2, run on Airflow 3.3.1 with Docker Compose on macOS. The original lab description follows unchanged below this section.

## Environment setup

I used the official Airflow 3.3.1 `docker-compose.yaml` with these edits:

- `_PIP_ADDITIONAL_REQUIREMENTS` set to `pandas scikit-learn flask requests`
- `LOAD_EXAMPLES` set to `false`
- `AIRFLOW_USERNAME` and `AIRFLOW_PASSWORD` environment variables mapped from the login in `.env`
- `./working_data` and `./model` mounted into the containers at `/opt/airflow/...`
- Port `5555` published on `airflow-worker` so the Flask dashboard is reachable from the host

Credentials live in `.env`, which is not committed.

## What I changed

### 1. Fixed the Flask dashboard authentication (`Flask_API.py`)
The original code called the Airflow REST API with HTTP basic auth. Airflow 3 rejects that with a `401`, so the dashboard always showed the failure page with `N/A` values, even after a successful run. The dashboard now requests a JWT from `POST /auth/token` and sends it as an `Authorization: Bearer` header. The token is cached and refreshed, and a request that gets a `401` is retried once with a fresh token.

### 2. Handled the port 5555 conflict (`Flask_API.py`)
Starting Flask a second time crashed the task with "address already in use". The task now checks whether port 5555 is taken and, if so, exits cleanly instead of failing, so reruns of `Airflow_Lab2` no longer produce a red Flask task.

### 3. Fixed the feature scaling (`model_development.py`)
The original preprocessing applied both `MinMaxScaler` and `StandardScaler` to the same columns, and also scaled the binary `Male` column. Now a single `StandardScaler` is applied to the four continuous columns only (`Daily Time Spent on Site`, `Age`, `Area Income`, `Daily Internet Usage`), and `Male` passes through unchanged. The fitted scaler is saved to `model/scaler.pkl` so new data can be scaled the same way as the training data.

### 4. Added full evaluation metrics (`model_development.py`)
`load_model` previously only printed accuracy. It now computes accuracy, precision, recall and F1 on the test set, prints them to the task log, and writes them to `model/metrics.json`.

### 5. Updated the dashboard (`Flask_API.py`, `success.html`, `failure.html`)
- The success page has a new "Model Performance" card that reads `metrics.json` and shows the model name, accuracy, F1, precision, recall, the number of test samples and the training time. If the file is missing, the page shows "Metrics not available yet" instead of failing.
- The failure page now shows the run state and the note explaining why (for example "Model pipeline is still in progress..." or an API error), which was previously collected but never displayed.

### 6. Personalized the DAG (`main.py`)
- Owner changed to my name, with a matching `owner_links` entry
- Tags changed from `example` to `mlops`, `lab2` and `logistic-regression`
- A description that explains what the pipeline does
- The instructor's email address was removed from the commented-out notification task. The email task stays disabled because it needs an SMTP connection configured in Airflow first.

## Results

Latest run, logistic regression on the 300-sample test split:

| Metric | Value |
| --- | --- |
| Accuracy | 0.97 |
| Precision | 0.9801 |
| Recall | 0.961 |
| F1 | 0.9705 |

All seven tasks in `Airflow_Lab2` succeed, and the dashboard at `http://localhost:5555` shows the success page with the metrics card.

## Notes for rerunning

Flask keeps running as a long-lived task, so it holds the code and templates it started with. After editing `Flask_API.py` or the templates, restart the worker so the new version loads, then trigger `Airflow_Lab2_Flask`:

```bash
docker compose restart airflow-worker
```

Plain reruns of `Airflow_Lab2` do not need a restart.

---

# Airflow Lab Instructions and Description

## Introduction
This document provides a detailed breakdown of the two DAGs (`Airflow_Lab2` and `Airflow_Lab2_Flask`) along with explanations of each function within the DAG files. These explanations aim to clarify the purpose and functionality of each task and function in the Airflow workflows.


Watch the tutorial video for this lab at [Airflow Lab2 Tutorial Video](https://youtu.be/LwBFOyfN5TY)

## Prerequisites
Before proceeding with this lab, ensure the following prerequisites are met:

- Basic understanding of Apache Airflow concepts.
- Apache Airflow installed and configured.
- Necessary Python packages installed, including Flask.


## Airflow Email Configuration

### Sign in with app passwords
Follow the instruction provided here: [link](https://support.google.com/accounts/answer/185833) and get your smtp password

### Adding SMTP Information to airflow.cfg

To configure Airflow to send emails, you need to add SMTP information to the `airflow.cfg` file. Follow these steps:

1. Locate the `airflow.cfg` file in your Airflow installation directory.
2. Open the file using a text editor.
3. Search for the `[smtp]` section in the configuration file.
4. Update the following parameters with your SMTP server information:
   - `smtp_host`: Hostname of the SMTP server.
   - `smtp_starttls`: Set it to `True` if your SMTP server uses TLS.
   - `smtp_ssl`: Set it to `True` if your SMTP server uses SSL.
   - `smtp_user`: Your SMTP username.
   - `smtp_password`: Your SMTP password.
   - `smtp_port`: Port number of the SMTP server (e.g., 587 for TLS, 465 for SSL).
5. Save the changes to the `airflow.cfg` file.

for our lab assuming you have a gmail account you can use the following setting:
   - smtp_host = smtp.gmail.com
   - smtp_starttls = True
   - smtp_ssl = False
   - smtp_user = YOUREMAIL@gmail.com
   - smtp_password = Enter your password generated above
   - smtp_port = 587
   - smtp_mail_from = YOUREMAIL@gmail.com
   - smtp_timeout = 30
   - smtp_retry_limit = 5

After updating the SMTP information, Airflow will use the configured SMTP server to send email notifications.


## DAG Structure

### `Airflow_Lab2`
This DAG orchestrates a machine learning pipeline and notification system. Let's break down each function within this DAG:

1. **`notify_success(context)` and `notify_failure(context)` Functions:**
   - These functions define email notifications for task success and failure, respectively. They utilize the `EmailOperator` to send emails with predefined content and subject to a specified recipient (in this case, `rey.mhmmd@gmail.com`).

2. **`default_args` Dictionary:**
   - This dictionary defines default arguments for the DAG, including the start date and the number of retries in case of task failure.

3. **`dag` Definition:**
   - This section creates the main DAG instance (`Airflow_Lab2`) with various parameters such as description, schedule interval, catchup behavior, and tags.
   
4. **`owner_task` BashOperator:**
   - This task echoes `1` and is assigned to an owner (`Ramin Mohammadi`). It represents a simple demonstration task with a linked owner.

5. **`send_email` EmailOperator:**
   - This task sends a notification email upon DAG completion. It utilizes the `notify_success` and `notify_failure` functions as callbacks for success and failure, respectively.

6. **PythonOperator Tasks:**
   - These tasks execute Python functions (`load_data`, `data_preprocessing`, `separate_data_outputs`, `build_model`, `load_model`) representing different stages of a machine learning pipeline. They perform data loading, preprocessing, model building, and model loading tasks.

7. **`TriggerDagRunOperator` Task:**
   - This task triggers the `Airflow_Lab2_Flask` DAG upon successful completion of the main DAG. It ensures that the Flask API is launched after the machine learning pipeline completes successfully.

### `Airflow_Lab2_Flask`
This DAG manages the Flask API's lifecycle and consists of the following function:

1. **`check_dag_status()` Function:**
   - This function queries the status of the last DAG run (`Airflow_Lab2`). It returns `True` if the DAG run was successful, and `False` otherwise.

2. **`handle_api_request()` Function:**
   - This function handles API requests and redirects users to `/success` or `/failure` routes based on the status of the last DAG run.

3. **Flask Routes and HTML Templates:**
   - The Flask routes (`/api`, `/success`, `/failure`) define endpoints for accessing the API and displaying success or failure pages. These routes render HTML templates (`success.html`, `failure.html`) with appropriate messages.

4. **`start_flask_app()` Function:**
   - This function starts the Flask server, enabling users to access the API endpoints.

5. **`start_flask_API` PythonOperator:**
   - This task executes the `start_flask_app()` function to initiate the Flask server. It represents the starting point for the Flask API's lifecycle.

## Conclusion
In this project, we've constructed a robust workflow using Apache Airflow to orchestrate a machine learning pipeline and manage a Flask API for monitoring purposes. The Airflow_Lab2 DAG coordinates various tasks, including data loading, preprocessing, model building, and email notification upon completion. By leveraging PythonOperators and BashOperator, we've encapsulated each step of the machine learning process, allowing for easy scalability and maintenance.

Additionally, the integration of email notifications enhances the workflow's visibility, providing stakeholders with timely updates on task success or failure. This ensures proactive monitoring and quick response to any issues that may arise during pipeline execution.

Furthermore, the Airflow_Lab2_Flask DAG facilitates the management of a Flask API, enabling users to access endpoints for checking the status of the machine learning pipeline. By querying the last DAG run status, the API delivers real-time feedback, empowering users to make informed decisions based on the pipeline's performance.

Overall, this project demonstrates the power of Apache Airflow in orchestrating complex workflows and integrating external systems seamlessly. By following the provided instructions and understanding the workflow's structure, users can leverage Airflow to streamline their machine learning pipelines and enhance operational efficiency.