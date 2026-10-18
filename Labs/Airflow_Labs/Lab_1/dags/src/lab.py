from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

DAGS_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = DAGS_DIR / "data"
WORK_DIR = Path(__file__).resolve().parents[2] / "working_data"
MODEL_DIR = WORK_DIR / "model"

FEATURES = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]
K_RANGE = range(1, 50)
KMEANS_KWARGS = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}


def load_data() -> str:
    """Read the raw CSV, check the needed columns exist, and stage a copy.
    Returns the file path, not the data itself."""
    df = pd.read_csv(DATA_DIR / "file.csv")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"file.csv is missing columns: {missing}")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = WORK_DIR / "raw.pkl"
    df.to_pickle(raw_path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns -> {raw_path}")
    return str(raw_path)


def data_preprocessing(raw_path: str) -> str:
    """Keep the clustering features, drop missing values, and scale to 0-1.
    Saves the fitted scaler so test data can be scaled the same way later."""
    df = pd.read_pickle(raw_path)
    features = df[FEATURES].dropna()
    print(f"Dropped {len(df) - len(features)} rows with missing values")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    data_path = WORK_DIR / "preprocessed.npy"
    np.save(data_path, scaled)
    print(f"Saved scaled data {scaled.shape} -> {data_path}")
    return str(data_path)

def compute_sse(data_path: str) -> list[float]:
    """Train K-Means for every k in K_RANGE and record each SSE (inertia).
    Returns a small list of numbers, which is safe to pass through XCom."""
    X = np.load(data_path)
    sse = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, **KMEANS_KWARGS).fit(X)
        sse.append(float(model.inertia_))
    print(f"Computed SSE for k = {K_RANGE.start} to {K_RANGE.stop - 1}")
    return sse


def find_optimal_k(sse: list[float]) -> int:
    """Use the elbow method to pick the number of clusters."""
    knee = KneeLocator(list(K_RANGE), sse, curve="convex", direction="decreasing")
    if knee.elbow is None:
        raise ValueError("No elbow found in the SSE curve; try a different K_RANGE.")
    k = int(knee.elbow)
    print(f"Optimal number of clusters: {k}")
    return k

def train_final_model(data_path: str, k: int) -> str:
    """Train the final K-Means using the optimal k, and save it."""
    X = np.load(data_path)
    model = KMeans(n_clusters=k, **KMEANS_KWARGS).fit(X)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"kmeans_k{k}.pkl"
    joblib.dump(model, model_path)

    sizes = np.bincount(model.labels_).tolist()
    print(f"Trained K-Means with k={k}. Cluster sizes: {sizes}")
    return str(model_path)


def predict_test(model_path: str) -> dict:
    """Scale test.csv with the saved scaler, predict a cluster for every row,
    and save the results to working_data/predictions.csv."""
    model = joblib.load(model_path)
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    test = pd.read_csv(DATA_DIR / "test.csv")
    missing = [c for c in FEATURES if c not in test.columns]
    if missing:
        raise ValueError(f"test.csv is missing columns: {missing}")
    test = test[FEATURES].dropna().copy()

    scaled = scaler.transform(test)
    out_of_range = int(((scaled < 0) | (scaled > 1)).any(axis=1).sum())
    if out_of_range:
        print(f"WARNING: {out_of_range} test row(s) have values outside the training range")

    test["cluster"] = model.predict(scaled)
    output_path = WORK_DIR / "predictions.csv"
    test.to_csv(output_path, index=False)

    counts = test["cluster"].value_counts().sort_index()
    print(f"Predicted {len(test)} row(s) -> {output_path}")
    return {
        "rows": len(test),
        "out_of_range_rows": out_of_range,
        "cluster_counts": {str(c): int(n) for c, n in counts.items()},
        "output_path": str(output_path),
    }