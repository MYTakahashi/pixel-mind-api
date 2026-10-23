# Bibliotecas
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class ModelMetrics(BaseModel):
    accuracy: float = Field(ge=0, le=1)
    f1_score: float = Field(ge=0, le=1)
    dataset: str = "MNIST - conjunto de teste"

class ModelPrediction(BaseModel):
    model: Literal["KNN", "Random Forest", "MLP"]
    predicted_digit: int = Field(ge=0, le=9)
    confidence: float = Field(ge=0, le=1)
    metrics: ModelMetrics

class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_digit: bool
    message: str
    processed_image_base64: str | None = None
    image_width: int = 28
    image_height: int = 28
    predictions: list[ModelPrediction] = Field(default_factory=list)
    note: str | None = None

class HealthResponse(BaseModel):
    status: Literal["ok", "unhealthy"]
    message: str
    models_loaded: dict[str, bool]
