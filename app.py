from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("model.pkl")

class InputData(BaseModel):
    merchant_group: str
    days_since_first_loan: int
    num_confirmed_payments_6m: int
    num_failed_payments_1y: int

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Decision Tree to predict debt default probability."}

@app.post("/predict")
def predict(data: InputData):
    X = pd.DataFrame([data.model_dump()])
    prediction = model.predict(X)[0]
    return {"prediction": float(prediction)}