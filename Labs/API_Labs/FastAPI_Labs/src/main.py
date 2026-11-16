from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

try:
    from .predict import load_model, predict_with_confidence
except ImportError:
    from predict import load_model, predict_with_confidence


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the server starts, instead of on every request.
    load_model()
    yield


app = FastAPI(
    title="Iris Species Classifier",
    description="Predicts the iris species from flower measurements using a decision tree.",
    version="1.1.0",
    lifespan=lifespan,
)


class IrisData(BaseModel):
    """
    Iris flower measurements in centimeters.
    Each value must be greater than 0 and at most 10 cm.
    """
    petal_length: float = Field(..., gt=0, le=10, description="Petal length in cm", examples=[1.4])
    sepal_length: float = Field(..., gt=0, le=10, description="Sepal length in cm", examples=[5.1])
    petal_width: float = Field(..., gt=0, le=10, description="Petal width in cm", examples=[0.2])
    sepal_width: float = Field(..., gt=0, le=10, description="Sepal width in cm", examples=[3.5])


class IrisResponse(BaseModel):
    response: int = Field(..., description="Predicted class: 0, 1 or 2")
    species: str = Field(..., description="Predicted species name")
    confidence: float = Field(..., description="Model confidence for the predicted class, from 0 to 1")


@app.get("/", status_code=status.HTTP_200_OK)
def health_ping():
    """Health check: confirms the API is running."""
    return {"status": "healthy"}


@app.post("/predict", response_model=IrisResponse)
def predict_iris(iris_features: IrisData):
    """
    Predict the iris flower species from its measurements.

    Values must be greater than 0 and at most 10 cm; otherwise the API returns 422.

    Example request:
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}

    Example response:
        {"response": 0, "species": "setosa", "confidence": 1.0}
    """
    try:
        # Feature order must match training: sepal length, sepal width, petal length, petal width.
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                     iris_features.petal_length, iris_features.petal_width]]
        class_id, species, confidence = predict_with_confidence(features)
        return IrisResponse(response=class_id, species=species, confidence=round(confidence, 4))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))