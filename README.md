# 🛡️ SENTINEL: NIGERIAN FRAUD DETECTION SYSTEM

*AI-Powered Real-Time Fraud Prevention for Nigerian E-Commerce & Fintech*

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2C2C2C?logo=xgboost&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini_AI-8E75B2?logo=googlegemini&logoColor=white)

---

## 📘 Table of Contents
- [The Sentinel Story](#-the-sentinel-story)
- [Overview](#-overview)
- [Model Performance](#-model-performance)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Use Cases](#-use-cases)

---

## 📖 The Sentinel Story

In the Nigerian fintech landscape, fraud isn't just a "data problem"—it’s a sophisticated, evolving challenge that targets specific local behaviors. Traditional, generic fraud models often fail here because they don't understand the nuance of a **USSD transfer at 2:00 AM**, the surge of activity during **Salary Week**, or the high-velocity nature of **Opay/Moniepoint** transactions.

**Sentinel** was built to bridge this gap. It doesn't just look at numbers; it looks at patterns. By combining "Surgical" feature engineering (like balance depletion ratios) with a high-performance ensemble of AI models, Sentinel protects both the bank's bottom line and the customer's peace of mind.

> "We built Sentinel to ensure that a legitimate customer in Lagos can spend their money without friction, while a fraudster in a dark room is blocked before the 'Send' button is even cold."


---

## 🚀 Overview

Sentinel is a production-grade Fraud Detection API that processes transactions in real-time. It uses a **Voting Ensemble (Random Forest + XGBoost + LightGBM)** to predict the probability of fraud with surgical precision.

### Why Sentinel is different:
* **Locally Contextual:** Engineered specifically for Nigerian banking triggers (USSD, BVN-linkage, Location-based spending).
* **Explainable AI:** Every "BLOCK" decision comes with a human-readable reason and a SHAP-based technical breakdown.
* **Business First:** Includes a built-in ROI calculator that translates model accuracy into **Naira Saved**.

---

## 📈 Model Performance

We don't just chase high accuracy; we chase **Precision**. In fraud detection, a "False Positive" means a frustrated customer who can't pay for their dinner. Sentinel is tuned to avoid that.

| Metric | Result | Meaning |
| :--- | :--- | :--- |
| **AUC-ROC** | **96.7%** | Excellent ability to distinguish between fraud and legit. |
| **Precision** | **99.8%** | Only 4 "False Alarms" out of 12,000+ transactions. |
| **Recall** | **87.9%** | We catch nearly 9 out of every 10 fraud attempts. |
| **False Positive Rate** | **0.03%** | Virtually zero friction for legitimate customers. |

**Net Business Impact (Test Set):**
* **Total Fraud Prevented:** ₦36,165,467.60
* **Customer Friction Cost:** -₦10,000.00
* **Net Profit:** **₦36,155,467.60**

---

## ✨ Features

* **⚡ Real-Time Scoring:** Sub-100ms inference using FastAPI and optimized model artifacts.
* **🧠 Surgical Engineering:** 26+ features including "Midnight High-Value Shock" and "Balance Depletion Ratio."
* **🔗 Redis-Powered Idempotency:** Prevents duplicate processing of transactions during network retries.
* **🚩 Automated Webhooks:** Sends instant POST alerts to your internal security systems for "High Risk" flags.
* **✍️ AI Executive Summaries:** Integrates **Google Gemini 2.0 Flash** to write professional daily fraud reports for management.
* **🔍 Explainability (SHAP):** Transparent AI that tells you *why* it flagged a transaction (e.g., "Amount is 30x higher than typical for Abuja").

---

## 🛠️ Technology Stack

* **Core:** Python 3.10+, FastAPI
* **ML/AI:** Scikit-Learn, XGBoost, LightGBM, SHAP, SMOTE (Imbalanced-learn)
* **Data:** Pandas, NumPy, Parquet
* **DevOps/Infra:** Redis (Caching), Uvicorn, Dotenv, Procfile (Heroku/Render ready)
* **Intelligence:** Google Gemini 2.0 (Generative Reports)

---

## 📂 Project Structure

```text
sentinel-fraud-detection/
├── data/
│   ├── nigerian_fraud_sent   # Processed features (Parquet)
│   └── nigerian_transactions # Raw transaction data
├── models/
│   ├── model_infos.json      # Metadata & Version tracking
│   ├── sentinel_ensemble     # Trained ensemble model artifact
│   └── sentinel_v2_Encoder   # Saved LabelEncoders
├── notebooks/
│   └── Experiments/
│       ├── 01_eda_and_features.ipynb # Behavioral Analysis
│       └── 02_model_training.ipynb   # Model calibration
├── src/
│   ├── generator.py          # Synthetic data generation logic
│   ├── logger.py             # Custom logging utility
│   └── predictor.py          # Inference wrapper logic
└── main.py               # FastAPI production server
├── core_function.py      # Decision logic & SHAP generators
├── .env                      # API keys & Configuration
├── Features.md               # Detailed feature documentation
├── Procfile                  # Deployment configuration
├── README.md                 # Main documentation
├── requirements.txt          # Python dependencies
├── sentinel_audit.log        # API transaction logs
└── test_sentinel.py          # API integration test script
---
```
## 🚀 Getting Started

### Prerequisites
* **Python 3.10+**: Core programming environment.
* **Redis Server**: Required for real-time idempotency and transaction caching (defaults to in-memory if Redis is unavailable).
* **Gemini API Key**: Required for generating the AI Fraud Summary reports.

### Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/sentinel-fraud-detection.git](https://github.com/yourusername/sentinel-fraud-detection.git)
   cd sentinel-fraud-detection
   
2. Set up the virtual environment:python -m venv venv
 ```bash
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
 ```
3. Install dependencies:
```
 pip install -r requirements.txt
```
4. Configure Environment Variables:
Create a .env file in the root directory:
```
    GEMINI_API_KEY=your_key_here
    REDIS_HOST=localhost
    REDIS_PORT=6379
 ```
## Usage:
1. Start the Sentinel Service:
```
 uvicorn api.main:app --reload
```
2. Access the Interactive Docs:
Open http://127.0.0.1:8000/docs in your browser to test endpoints via Swagger UI.
### API Testing
​Run the automated test script to simulate a high-risk transaction and verify the SHAP explanation engine:
```
python test_sentinel.py
```
 # Use Cases
* Fintech Wallets: Block "Midnight Sweeper" attacks where compromised accounts are drained via USSD while the owner sleeps.
* E-Commerce Checkouts: Flag high-value orders that deviate significantly from a user's 30-day spending baseline.
* Agency Banking: Monitor POS terminals for unusual transaction velocity or large withdrawals in high-risk geographical zones. License 
#### This project is licensed under the MIT License - see the LICENSE file for details.
​Sentinel: Protecting the pulse of Nigerian Digital Finance.
