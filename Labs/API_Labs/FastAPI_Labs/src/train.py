from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

try:
    from .data import load_data, split_data
except ImportError:
    from data import load_data, split_data

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "iris_model.pkl"


def fit_model(X_train, y_train):
    """
    Train a Decision Tree Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    Returns:
        DecisionTreeClassifier: The trained model.
    """
    dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=12)
    dt_classifier.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dt_classifier, MODEL_PATH)
    return dt_classifier


def evaluate_model(model, X_test, y_test):
    """
    Compute the accuracy of the model on the test set.
    Args:
        model: A trained classifier.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test target values.
    Returns:
        float: Test accuracy between 0 and 1.
    """
    return accuracy_score(y_test, model.predict(X_test))


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Trained on {len(y_train)} samples, tested on {len(y_test)} samples")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Model saved to {MODEL_PATH}")