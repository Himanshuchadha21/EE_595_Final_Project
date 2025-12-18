from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Medical Insurance Cost Predictor", version="1.0.0")
model = joblib.load("model.joblib")

class PredictRequest(BaseModel):
    age: int = Field(..., ge=0, le=40)
    sex: str = Field(..., examples=["male"])
    bmi: float = Field(..., ge=10, le=80)
    children: int = Field(..., ge=0, le=15)
    smoker: str = Field(..., examples=["yes"])
    region: str = Field(..., examples=["southeast"])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.model_dump()])
    pred = model.predict(df)[0]
    return {"predicted_charges": float(pred)}
