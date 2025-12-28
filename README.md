# Sentinel: Fraud Detection for Nigerian Fintech

**Sentinel** is an ensemble machine learning system designed to protect transactions from high-velocity fraud.

## 🚀 Performance Metrics
- **Net Business Value:** ₦179,520,522.55 (Saved Fraud - Friction Cost)
- **Precision:** 99%
- **Recall:** 84%

## 🛠 Tech Stack
- **Models:** Logistic Regression + XGBoost + LightGBM (Voting Ensemble)
- **API:** FastAPI
- **Security:** Tiered Response (Allow/Review/Block)

## 🏃 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Start API: `uvicorn api.main:app --reload`
3. Test docs: Go to `http://127.0.0.1:8000/docs`