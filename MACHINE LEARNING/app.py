from fastapi import FastAPI
import pickle
import numpy as np


app = FastAPI()

with open("pipeline_v1.bin", "rb") as f_in:
    model = pickle.load(f_in)

@app.get("/")
def root():
    return {"message": "Lead conversion prediction API is running!"}

@app.post("/predict")
def predict(client: dict):
    X = [client]
    proba = model.predict_proba(X)[0, 1]
    return {"conversion_probability": float(proba)}
