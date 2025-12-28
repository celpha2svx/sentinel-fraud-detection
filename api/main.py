from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Sentinel Fraud Detection Service")

# 1. Load the Gold Package
artifact = joblib.load("models/sentinel_v1_Encoder.joblib")
model = artifact['model']
encoders = artifact['encoders']
feature_names = artifact['feature_names']


class Transaction(BaseModel):
    amount: float
    user_avg_amount: float
    avg_tx_amount: float
    user_balance: float
    channel: str
    location: str
    sender_bank: str
    hour: int
    day_of_week: int
    is_ussd: int
    tx_count_24h: int
    total_spend_24h: float


@app.post("/predict")
async def predict(txn: Transaction):
    input_data = pd.DataFrame([txn.model_dump()])

    # 2. FEATURE ENGINEERING
    input_data['amount_vs_avg_ratio'] = input_data['amount'] / input_data['user_avg_amount']
    input_data['is_midnight'] = input_data['hour'].apply(lambda x: 1 if 0 <= x <= 5 else 0)

    bins = [0, 10000, 50000, 200000, 1000000, float('inf')]
    labels = ['Micro', 'Small', 'Medium', 'High', 'Whale']
    input_data['amount_band'] = pd.cut(input_data['amount'], bins=bins, labels=labels)

    # 3. Apply Encoders
    for col, le in encoders.items():
        if col in input_data.columns:
            try:
                input_data[col] = le.transform(input_data[col].astype(str))
            except ValueError:
                input_data[col] = -1


    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0

    final_input = input_data[feature_names]

    # 4. Predict
    proba = float(model.predict_proba(final_input)[0][1])

    if proba >= 0.8:
        action, risk = "BLOCK", "High"
    elif proba >= 0.5:
        action, risk = "REVIEW", "Medium"
    else:
        action, risk = "ALLOW", "Low"

    return {
        "fraud_probability": round(proba, 4),
        "risk_level": risk,
        "action": action
    }