from fastapi import FastAPI, Header, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import joblib
import redis
import uvicorn
import shap
import time
from datetime import datetime, timedelta
import logging
import uuid
import json
import os
from dotenv import load_dotenv

# Import functions
from api.core_function import (
    Transaction, RiskLevel, Action,
    process_single_txn, generate_shap_explanation
)

load_dotenv()


# LOGGING CONFIGURATION
logging.basicConfig(
    filename='sentinel_audit.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)


# APP INITIALIZATION
app = FastAPI(
    title="Sentinel Fraud Detection Service",
    version="3.0.0",
    description="Production-grade fraud detection with 96.7% AUC, Redis caching, webhooks, and AI reports"
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# LOAD MODEL ARTIFACTS
try:
    model_path = "models/sentinel_v3_PRODUCTION.joblib"
    artifact = joblib.load(model_path)

    model = artifact['model']
    encoders = artifact['encoders']
    feature_names = artifact['feature_names']
    thresholds = artifact['threshold_config']

    print(f" Model loaded successfully from {model_path}")
    print(f"Features: {len(feature_names)}")
    print(f"Thresholds: {thresholds}")

except Exception as e:
    print(f"CRITICAL: Failed to load model: {e}")
    raise


# SHAP EXPLAINER SETUP
try:
    if hasattr(model, 'named_estimators_'):
        xgb_model = model.named_estimators_['xgb']
    else:
        xgb_model = model

    explainer = shap.TreeExplainer(xgb_model)
    print("SHAP explainer initialized")
except Exception as e:
    print(f"SHAP explainer failed: {e}")
    explainer = None


# REDIS CONFIGURATION
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0,
        decode_responses=True,
        socket_timeout=5
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("Redis connected successfully")
except Exception as e:
    print(f"Redis not available: {e}. Using in-memory fallback.")
    REDIS_AVAILABLE = False
    MEMORY_CACHE = {}

# CACHE HELPER FUNCTIONS
def cache_set(key: str, value: dict, ttl_seconds: int = 86400):
    """Store data in cache with TTL (Time To Live)."""
    try:
        if REDIS_AVAILABLE:
            redis_client.setex(key, ttl_seconds, json.dumps(value))
        else:
            MEMORY_CACHE[key] = {
                'data': value,
                'expires_at': datetime.now() + timedelta(seconds=ttl_seconds)
            }
    except Exception as e:
        logging.error(f"Cache set error: {e}")


def cache_get(key: str) -> Optional[dict]:
    """Retrieve data from cache."""
    try:
        if REDIS_AVAILABLE:
            data = redis_client.get(key)
            return json.loads(data) if data else None
        else:
            cached = MEMORY_CACHE.get(key)
            if cached and datetime.now() < cached['expires_at']:
                return cached['data']
            elif cached:
                del MEMORY_CACHE[key]
            return None
    except Exception as e:
        logging.error(f"Cache get error: {e}")
        return None


def cache_delete(key: str):
    """Delete data from cache."""
    try:
        if REDIS_AVAILABLE:
            redis_client.delete(key)
        else:
            MEMORY_CACHE.pop(key, None)
    except Exception as e:
        logging.error(f"Cache delete error: {e}")


def cache_clear_pattern(pattern: str):
    """Clear all keys matching pattern."""
    try:
        if REDIS_AVAILABLE:
            keys = redis_client.keys(pattern)
            if keys:
                redis_client.delete(*keys)
        else:
            keys_to_delete = [k for k in MEMORY_CACHE.keys() if pattern.replace('*', '') in k]
            for k in keys_to_delete:
                del MEMORY_CACHE[k]
    except Exception as e:
        logging.error(f"Cache clear error: {e}")


# GLOBAL METRICS
START_TIME = time.time()
METRICS = {
    'PREDICTION_COUNT': 0,
    'FRAUD_COUNT': 0,
    'TOTAL_VALUE_SAVED': 0.0
}



# WEBHOOK NOTIFICATION SYSTEM
async def send_webhook_notification(result: dict, webhook_url: str):
    """
    Sends webhook notification for high-risk transactions.
    Actually sends HTTP POST to the bank's system!
    """
    if result['data']['risk_level'] == RiskLevel.HIGH:
        import httpx

        message = {
            "alert_type": "HIGH_RISK_FRAUD",
            "request_id": result['request_id'],
            "amount": result['data']['amount'],
            "fraud_probability": result['data']['fraud_probability'],
            "recommended_action": result['data']['recommended_action'],
            "explanation": result['data']['explanation'],
            "timestamp": result['meta']['timestamp'],
            "channel": result['data']['channel'],
            "location": result['data']['location'],
            "bvn_linked": result['data']['bvn_linked'],
            "device_changed": result['data']['device_changed']
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(webhook_url, json=message)

                if response.status_code == 200:
                    logging.info(f"Webhook sent to {webhook_url}")
                    print(f"[WEBHOOK] Alert sent successfully")
                else:
                    logging.warning(f"Webhook failed: {response.status_code}")
                    print(f"[WEBHOOK]Failed: {response.status_code}")

        except Exception as e:
            logging.error(f"Webhook error: {e}")
            print(f"[WEBHOOK]Error: {e}")


# AI FRAUD SUMMARY REPORT (GEMINI)
async def generate_ai_fraud_report(
        start_date: str,
        end_date: str,
        fraud_transactions: List[dict]
) -> dict:
    """
    Uses Gemini 2.0 Flash to generate a manager-friendly fraud summary report.
    Only called when /report/fraud-summary is requested.
    """
    import httpx

    gemini_api_key = os.getenv('GEMINI_API_KEY')

    if not gemini_api_key:
        return {
            "error": "Gemini API key not configured. Add GEMINI_API_KEY to .env file"
        }

    # data summary
    total_fraud_amount = sum(tx['data']['amount'] for tx in fraud_transactions)
    avg_fraud_amount = total_fraud_amount / len(fraud_transactions) if fraud_transactions else 0

    # Group by channel
    channel_breakdown = {}
    for tx in fraud_transactions:
        channel = tx['data'].get('channel', 'Unknown')
        channel_breakdown[channel] = channel_breakdown.get(channel, 0) + 1

    # prompt for Gemini
    prompt = f"""
You are a fraud analyst writing a daily summary report for a Nigerian bank manager.

**Data Summary:**
- Report Period: {start_date} to {end_date}
- Total High-Risk Transactions Detected: {len(fraud_transactions)}
- Total Amount Saved: ₦{total_fraud_amount:,.2f}
- Average Fraud Amount: ₦{avg_fraud_amount:,.2f}

**Channel Breakdown:**
{json.dumps(channel_breakdown, indent=2)}

**Sample High-Risk Transactions:**
{json.dumps(fraud_transactions[:5], indent=2)}

**Please write a professional fraud summary report that includes:**
1. Executive summary (2-3 sentences)
2. Key trends and patterns observed
3. Top risk factors identified
4. Recommended actions for the fraud team
5. Notable high-value cases

Keep it concise, professional, and actionable for a banking executive.
"""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent",
                headers={"Content-Type": "application/json"},
                params={"key": gemini_api_key},
                json={
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 1500
                    }
                }
            )

            if response.status_code == 200:
                data = response.json()
                report_text = data['candidates'][0]['content']['parts'][0]['text']

                return {
                    "status": "success",
                    "report": report_text,
                    "metadata": {
                        "period": f"{start_date} to {end_date}",
                        "total_fraud_detected": len(fraud_transactions),
                        "total_amount_saved": f"₦{total_fraud_amount:,.2f}",
                        "generated_at": datetime.now().isoformat(),
                        "model": "gemini-2.0-flash-exp"
                    }
                }
            else:
                return {
                    "error": f"Gemini API error: {response.status_code}",
                    "details": response.text
                }

    except Exception as e:
        logging.error(f"AI report generation error: {e}")
        return {"error": str(e)}


# API ENDPOINTS
@app.get("/")
def root():
    """API documentation and welcome message."""
    return {
        "service": "Sentinel Fraud Detection API",
        "version": "3.0.0",
        "description": "Production-grade fraud detection for Nigerian financial institutions",
        "model_performance": {
            "auc_roc": 0.9679,
            "precision": 0.998,
            "recall": 0.879,
            "false_positive_rate": 0.0003
        },
        "features": [
            "96.7% AUC-ROC",
            "99.8% Precision (almost no false alarms)",
            "Redis caching (24h TTL)",
            "Idempotency keys for network resilience",
            "Real-time webhook notifications",
            "Simulation mode for testing",
            "Batch processing",
            "SHAP-based explanations",
            "Real-time metrics",
            "AI-powered fraud summaries (Gemini)"
        ],
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "batch": "POST /batch",
            "explain": "POST /explain",
            "metrics": "GET /metrics",
            "ai_report": "POST /report/fraud-summary",
            "cache_clear": "DELETE /cache/clear"
        },
        "documentation": "/docs",
        "status": "operational"
    }


@app.get("/health")
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "uptime_seconds": int(time.time() - START_TIME),
        "version": "3.0.0",
        "total_predictions_served": METRICS['PREDICTION_COUNT'],
        "model_loaded": model is not None,
        "redis_available": REDIS_AVAILABLE,
        "shap_available": explainer is not None
    }


@app.post("/predict")
async def predict(
        txn: Transaction,
        background_tasks: BackgroundTasks,
        x_idempotency_key: Optional[str] = Header(None),
        x_simulation: Optional[str] = Header(None),
        x_webhook_url: Optional[str] = Header(None)
):
    """
    Single transaction prediction endpoint.

    Headers:
    - X-Idempotency-Key: Prevents duplicate processing (UUID recommended)
    - X-Simulation: Set to 'true' for testing without affecting metrics
    - X-Webhook-Url: URL to receive high-risk alerts (actually sends HTTP POST!)
    """
    # Handle simulation mode
    is_simulation = x_simulation and x_simulation.lower() == 'true'

    # Handle idempotency
    if x_idempotency_key:
        cache_key = f"idempotency:{x_idempotency_key}"
        cached_response = cache_get(cache_key)

        if cached_response:
            print(f"[IDEMPOTENT] Returning cached response for key: {x_idempotency_key}")
            return cached_response

    # Generate unique request ID
    request_id = str(uuid.uuid4())

    # Process transaction
    result = process_single_txn(
        txn=txn,
        request_id=request_id,
        model=model,
        encoders=encoders,
        feature_names=feature_names,
        is_simulation=is_simulation,
        metrics=METRICS if not is_simulation else None
    )

    # Store transaction data in cache for /explain endpoint (24h TTL)
    if not is_simulation:
        cache_key = f"transaction:{request_id}"
        cache_set(cache_key, result['_internal'], ttl_seconds=86400)

    # Remove internal data from response
    result_clean = {k: v for k, v in result.items() if k != '_internal'}

    # Cache for idempotency - 7 days TTL
    if x_idempotency_key:
        cache_key = f"idempotency:{x_idempotency_key}"
        cache_set(cache_key, result_clean, ttl_seconds=604800)

    # Send webhook notification in background (if high risk and webhook provided)
    if x_webhook_url and not is_simulation and result['data']['risk_level'] == RiskLevel.HIGH:
        background_tasks.add_task(send_webhook_notification, result_clean, x_webhook_url)

    return result_clean


@app.post("/batch")
async def predict_batch(
        transactions: List[Transaction],
        background_tasks: BackgroundTasks,
        x_simulation: Optional[str] = Header(None),
        x_webhook_url: Optional[str] = Header(None)
):
    """
    Batch prediction endpoint for processing multiple transactions.
    """
    is_simulation = x_simulation and x_simulation.lower() == 'true'

    results = []
    high_risk_count = 0

    for txn in transactions:
        request_id = str(uuid.uuid4())
        result = process_single_txn(
            txn=txn,
            request_id=request_id,
            model=model,
            encoders=encoders,
            feature_names=feature_names,
            is_simulation=is_simulation,
            metrics=METRICS if not is_simulation else None
        )

        # Store in cache for explain endpoint
        if not is_simulation:
            cache_key = f"transaction:{request_id}"
            cache_set(cache_key, result['_internal'], ttl_seconds=86400)

        # Remove internal data
        result_clean = {k: v for k, v in result.items() if k != '_internal'}
        results.append(result_clean)

        if result['data']['risk_level'] == RiskLevel.HIGH:
            high_risk_count += 1

            # Send webhook for each high-risk transaction
            if x_webhook_url and not is_simulation:
                background_tasks.add_task(send_webhook_notification, result_clean, x_webhook_url)

    return {
        "status": "success",
        "batch_size": len(transactions),
        "high_risk_count": high_risk_count,
        "results": results,
        "simulation": is_simulation,
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "version": "3.0.0"
        }
    }


@app.post("/explain")
async def explain_transaction(request_id: str):
    """
    Provides SHAP-based explanation for a specific transaction.
    Shows which features drove the fraud score.
    """
    if not explainer:
        raise HTTPException(
            status_code=503,
            detail="SHAP explainer not available"
        )

    cache_key = f"transaction:{request_id}"
    cached_data = cache_get(cache_key)

    if not cached_data:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {request_id} not found. It may have expired from cache (24h TTL)."
        )

    # Generate SHAP explanation
    shap_result = generate_shap_explanation(
        features_dict=cached_data['features'],
        feature_names=feature_names,
        explainer=explainer
    )

    return {
        "status": "success",
        "request_id": request_id,
        "transaction_details": cached_data['transaction'],
        **shap_result,
        "meta": {
            "analyzed_at": datetime.now().isoformat()
        }
    }


@app.get("/metrics")
def get_metrics():
    """System metrics and performance statistics."""
    uptime_seconds = time.time() - START_TIME
    uptime_hours = uptime_seconds / 3600

    fraud_rate = (METRICS['FRAUD_COUNT'] / METRICS['PREDICTION_COUNT'] * 100) if METRICS['PREDICTION_COUNT'] > 0 else 0

    return {
        "status": "operational",
        "uptime": {
            "seconds": int(uptime_seconds),
            "hours": round(uptime_hours, 2),
            "human_readable": f"{int(uptime_hours)}h {int((uptime_seconds % 3600) / 60)}m"
        },
        "predictions": {
            "total_requests": METRICS['PREDICTION_COUNT'],
            "fraud_detected": METRICS['FRAUD_COUNT'],
            "fraud_rate_percent": round(fraud_rate, 2)
        },
        "financial_impact": {
            "total_value_saved": f"₦{METRICS['TOTAL_VALUE_SAVED']:,.2f}",
            "total_value_saved_raw": METRICS['TOTAL_VALUE_SAVED']
        },
        "model_performance": {
            "auc_roc": 0.9679,
            "precision": 0.998,
            "recall": 0.879,
            "false_positive_rate": 0.0003
        },
        "system_health": {
            "redis_available": REDIS_AVAILABLE,
            "shap_available": explainer is not None,
            "status": "Optimal"
        },
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "version": "3.0.0"
        }
    }


@app.post("/report/fraud-summary")
async def fraud_summary_report(
        start_date: str,
        end_date: str
):
    """
    Generate AI-powered fraud summary report using Gemini 2.0 Flash.
    Only called on-demand by managers.

    Example: POST /report/fraud-summary?start_date=2024-01-01&end_date=2024-01-31
    """
    fraud_transactions = []

    # Generate AI report
    report = await generate_ai_fraud_report(start_date, end_date, fraud_transactions)

    return report


@app.delete("/cache/clear")
def clear_cache(cache_type: str = "all"):
    """
    Administrative endpoint to clear caches.
    Options: 'idempotency', 'transactions', 'all'
    """
    if cache_type in ["idempotency", "all"]:
        cache_clear_pattern("idempotency:*")

    if cache_type in ["transactions", "all"]:
        cache_clear_pattern("transaction:*")

    return {
        "status": "success",
        "message": f"Cleared {cache_type} cache(s)",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)