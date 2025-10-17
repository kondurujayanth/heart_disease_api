from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Initialize FastAPI app
app = FastAPI(title="Heart Disease Prediction API")

# Load your trained model
model = joblib.load("model")

# Define input schema
class HeartData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is live!"}

@app.post("/predict")
def predict(data: HeartData):
    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}
