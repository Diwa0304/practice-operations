# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pydantic import create_model
import os
from google.cloud import storage


app = FastAPI(title="Fraud Detection API")

@app.on_event("startup")
async def load_model_from_gcs():
    global model
    
    # Load model from GCS
    BUCKET_NAME = "mlops-week-4-tough-study-473007-a5"
    GCS_DESTINATION_PATH = "models_oppe/final_logistic_regression.pkl"
    LOCAL_PATH = "temp/best_transaction_model.pkl"
    
    # Ensure local directory exists
    os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(GCS_DESTINATION_PATH)
    
    # Download blob to local file
    blob.download_to_filename(LOCAL_PATH)
    
    # Load the model
    model = joblib.load(LOCAL_PATH)

# Input schema
columns = ['Unnamed: 0', 'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 
       'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
       'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 
       'Amount', 'Class']
feature_columns = [c for c in columns if c not in ["Class"]]
TransactionInput = create_model(
"TransactionInput",
**{col: (float, ...) for col in feature_columns}
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Transaction Classifier API!"}

@app.post("/predict/")
def predict_transaction(data: TransactionInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {"predicted_class": int(prediction)}
