from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "iris_model.pkl"

SPECIES = ["setosa", "versicolor", "virginica"]

_model = None


def load_model():
    """
    Load the trained model once and keep it in memory for later requests.
    Returns:
        The trained classifier.
    Raises:
        FileNotFoundError: If the model file does not exist yet.
    """
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Run train.py first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (array-like): Input rows of [sepal_length, sepal_width, petal_length, petal_width].
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    return load_model().predict(X)


def predict_with_confidence(X):
    """
    Predict the class, species name and confidence for a single input row.
    Args:
        X (array-like): One row of [sepal_length, sepal_width, petal_length, petal_width].
    Returns:
        tuple: (class_id, species_name, confidence between 0 and 1)
    """
    probabilities = load_model().predict_proba(X)[0]
    class_id = int(probabilities.argmax())
    return class_id, SPECIES[class_id], float(probabilities[class_id])