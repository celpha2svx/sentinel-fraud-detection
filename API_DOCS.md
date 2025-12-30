# 🔌 Sentinel API Documentation (v3.0.0)

Welcome to the **Sentinel Fraud Detection API**. This service provides real-time transaction scoring, explainable AI insights, and automated fraud reporting.

---

## 🏗️ System Architecture

Sentinel is designed for high availability and "fail-safe" financial operations.

* **FastAPI Core**: Handles high-concurrency requests with asynchronous processing.
* **Redis Layer**: Manages 24-hour transaction caching and 7-day idempotency storage.
* **Gemini AI**: Generative layer for natural language executive summaries.
* **SHAP Engine**: Provides mathematical transparency for every decision.



---

## 🔐 Core Headers

Every request to the `/predict` endpoint should include these headers for production stability:

| Header | Type | Description |
| :--- | :--- | :--- |
| `X-Idempotency-Key` | UUID | **Required.** Prevents double-processing if the network retries a request. |
| `X-Simulation` | Boolean | If `true`, the API returns a score but does not update financial metrics. |
| `X-Webhook-Url` | URL | Optional URL to receive a POST alert for `HIGH` risk transactions. |

---

## 📡 API Endpoints

### 1. Single Transaction Prediction
`POST /predict`

**Request Body:**
```json
{
  "amount": 450000.0,
  "user_avg_amount": 15000.0,
  "user_balance": 500000.0,
  "channel": "USSD",
  "location": "Lagos",
  "sender_bank": "Zenith",
  "bvn_linked": 0,
  "device_changed": 1,
  "failed_attempts_24h": 3,
  "tx_count_1h": 5,
  "hour": 2,
  "is_midnight_high_value": 1,
  "pct_balance_withdrawn": 90.0
}
```
### Response:
```
{
  "request_id": "75040b7d-058e-4fed-8199-7ffa94045250",
  "status": "success",
  "data": {
    "fraud_probability": 0.9895,
    "risk_level": "High",
    "recommended_action": "BLOCK",
    "explanation": "NEW DEVICE DETECTED | High-value transaction without BVN..."
  }
}
```
### 2. AI Fraud Summary
`POST /report/fraud-summary`

Generates a manager-friendly report using **Gemini 2.0 Flash**.
* **Query Params**: `start_date` (YYYY-MM-DD), `end_date` (YYYY-MM-DD)
* **Functionality**: Aggregates all transactions marked as `High Risk` within the date range and sends a structured prompt to Gemini to summarize patterns and financial impact.
* **Output**: A professional executive summary containing:
    1. Executive summary of risk levels.
    2. Observed trends (e.g., "Increase in USSD fraud in Lagos").
    3. Recommended team actions.

### 3. Explanation Engine
`POST /explain`

Provides "Glass Box" transparency for machine learning decisions.
* **Path Params**: `request_id`
* **Mechanism**: Uses **SHAP (SHapley Additive exPlanations)** to calculate the contribution of each feature to the final fraud score.
* **Output**: 
    - `top_fraud_drivers`: Features that pushed the score higher.
    - `top_legitimacy_drivers`: Features that made the transaction look safe.

---

## 🛠️ Error Handling & Status Codes

| Code | Meaning | Action |
| :--- | :--- | :--- |
| **200** | Success | Process the `recommended_action` provided in the data block. |
| **404** | Not Found | Transaction ID has expired from Redis cache (24h TTL). |
| **422** | Validation Error | JSON schema is invalid. Ensure all 26 features are sent. |
| **503** | Service Unavailable | Redis is disconnected or the ML model file is missing. |

---

## 📈 System Health & Metrics

Sentinel includes real-time telemetry to track ROI and system stability.

* `GET /health`: Liveness probe for monitoring tools (Check uptime and Redis status).
* `GET /metrics`: Returns a live dashboard of:
    - **Total Value Saved**: Cumulative ₦ saved from blocked fraud.
    - **Fraud Rate**: Percentage of requests flagged as High Risk.
    - **Model Performance**: Current AUC-ROC and Precision stats.

---

## 📝 Audit Logs

The system maintains a tamper-evident audit trail in `sentinel_audit.log`. Every prediction is logged for compliance:
`2025-12-30 10:42:36 | INFO | ID: 75040b... | AMT: 450000.0 | PROBA: 0.9895 | ACT: BLOCK`

---

**End of API Documentation**