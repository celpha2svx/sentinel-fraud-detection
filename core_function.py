# api/core_function.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime
import logging
from enum import Enum
from pydantic import BaseModel, Field



# ENUMS & MODELS
class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class Action(str, Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class Transaction(BaseModel):
    """Transaction input schema with all 26 features"""
    # Basic transaction info
    amount: float = Field(..., gt=0, description="Transaction amount in Naira")
    user_avg_amount: float = Field(..., ge=0)
    user_balance: float = Field(..., ge=0)

    # Categorical features
    channel: str = Field(..., description="Mobile App, POS, or USSD")
    location: str = Field(..., description="Lagos, Abuja, Kano, Ibadan, PH")
    sender_bank: str = Field(..., description="Access, GTBank, Kuda, Moniepoint, OPay, Zenith")

    # Risk indicators
    bvn_linked: int = Field(1, ge=0, le=1, description="1 if BVN verified, 0 otherwise")
    device_changed: int = Field(0, ge=0, le=1, description="1 if new device detected")
    failed_attempts_24h: int = Field(0, ge=0, description="Failed login attempts in last 24h")

    # Velocity features
    tx_count_1h: int = Field(1, ge=0, description="Transaction count in last hour")
    tx_count_24h: float = Field(0.0, ge=0, description="Transaction count in last 24h")

    # Time features
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Sunday)")
    day: int = Field(..., ge=1, le=31, description="Day of month")
    is_midnight: int = Field(0, ge=0, le=1, description="1 if between midnight-5AM")
    is_salary_window: int = Field(0, ge=0, le=1, description="1 if salary payment period")
    is_ussd: int = Field(0, ge=0, le=1, description="1 if USSD channel")

    # Spend patterns
    total_spend_24h: float = Field(0.0, ge=0, description="Total spending in last 24h")
    amt_to_user_avg_ratio: float = Field(1.0, ge=0, description="Amount vs user average ratio")
    pct_of_balance: float = Field(1.0, ge=0, description="Transaction as % of balance")

    # Location patterns
    location_avg_30d: Optional[float] = Field(None, ge=0, description="30-day location average")
    amt_to_location_avg_ratio: float = Field(1.0, ge=0, description="Amount vs location average")

    # Additional velocity
    channel_velocity_1h: float = Field(0.0, ge=0, description="Channel transactions per hour")
    pct_balance_withdrawn: float = Field(1.0, ge=0, description="Percentage of balance withdrawn")
    tx_per_hour: float = Field(0.5, ge=0, description="Average transactions per hour")
    is_midnight_high_value: int = Field(0, ge=0, le=1, description="1 if high value at midnight")



# FEATURE ENGINEERING
def apply_encoders(input_data: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Apply label encoders to categorical features.
    Handles unseen categories gracefully by assigning them to a default value.
    """
    for col, le in encoders.items():
        if col in input_data.columns:
            try:
                input_data[col] = le.transform(input_data[col].astype(str))
            except ValueError as e:
                logging.warning(f"Unseen category in {col}: {input_data[col].values[0]}")
                input_data[col] = 0

    return input_data


def ensure_features(input_data: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Ensure all required features exist in the correct order.
    Adds missing features with default value 0.
    """
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0

    return input_data[feature_names]



# EXPLANATION GENERATION
def generate_human_explanation(proba: float, txn: Transaction) -> str:
    """
    Translates ML model signals into banker-friendly language.
    Runs FAST (no AI calls) for real-time API responses.
    """
    reasons = []

    # Rule 1: Device security issues
    if txn.device_changed == 1:
        reasons.append("NEW DEVICE DETECTED - possible account takeover")

    # Rule 2: BVN verification
    if txn.bvn_linked == 0 and txn.amount > 50000:
        reasons.append("High-value transaction without BVN verification")

    # Rule 3: Failed login attempts
    if txn.failed_attempts_24h > 2:
        reasons.append(f"{txn.failed_attempts_24h} failed login attempts in 24h")

    # Rule 4: Midnight high-value
    if txn.is_midnight_high_value == 1:
        reasons.append("High-value transaction during high-risk hours (midnight-5AM)")

    # Rule 5: Amount deviation
    if txn.amt_to_user_avg_ratio > 5:
        reasons.append(f"Transaction is {txn.amt_to_user_avg_ratio:.1f}x higher than user average")

    # Rule 6: Balance drain
    if txn.pct_balance_withdrawn > 80:
        reasons.append(f" Attempting to withdraw {txn.pct_balance_withdrawn:.0f}% of account balance")

    # Rule 7: Velocity anomaly
    if txn.tx_count_24h > 20:
        reasons.append(f"Unusual velocity: {int(txn.tx_count_24h)} transactions in 24h")

    # Rule 8: USSD high value
    if txn.is_ussd == 1 and txn.amount > 100000:
        reasons.append("Large USSD transfer (₦100k+) - uncommon pattern")

    # Rule 9: Location anomaly
    if txn.amt_to_location_avg_ratio > 10:
        reasons.append(f"Amount is {txn.amt_to_location_avg_ratio:.1f}x higher than typical for this location")

    # Fallback
    if not reasons:
        reasons.append("Overall behavioral pattern shows deviation from historical baseline")

    return " | ".join(reasons)



# DECISION LOGIC
def determine_action(proba: float) -> Tuple[Action, RiskLevel]:
    """
    Maps fraud probability to actionable decisions.
    Thresholds based on your model's optimal performance:
    - 0.8+ = BLOCK (99.8% precision)
    - 0.5-0.8 = REVIEW (manual investigation)
    - <0.5 = ALLOW
    """
    if proba >= 0.8:
        return Action.BLOCK, RiskLevel.HIGH
    elif proba >= 0.5:
        return Action.REVIEW, RiskLevel.MEDIUM
    else:
        return Action.ALLOW, RiskLevel.LOW



# CORE PROCESSING FUNCTION
def process_single_txn(
        txn: Transaction,
        request_id: str,
        model,
        encoders: dict,
        feature_names: list,
        is_simulation: bool = False,
        metrics: Optional[dict] = None
) -> dict:
    """
    Core processing function for a single transaction.
    Powers both /predict and /batch endpoints.

    Args:
        txn: Transaction object with all required fields
        request_id: Unique identifier for this request
        model: Trained ML model
        encoders: Dictionary of label encoders
        feature_names: List of feature names in correct order
        is_simulation: If True, doesn't update metrics
        metrics: Dictionary to update with metrics (passed by reference)

    Returns:
        Dictionary with prediction results
    """
    try:
        # Auto-calculate location_avg_30d if not provided
        if txn.location_avg_30d is None:
            txn.location_avg_30d = txn.user_avg_amount * 1.2

        # Convert to DataFrame
        input_data = pd.DataFrame([txn.model_dump()])

        # Apply encoders to categorical features
        input_data = apply_encoders(input_data, encoders)

        # Ensure all features exist in correct order
        final_input = ensure_features(input_data, feature_names)

        # Debug logging
        print(f"DEBUG: Final input shape: {final_input.shape}")
        print(f"DEBUG: Final input values: {final_input.values[0][:5]}...")  # First 5 features

        # Model prediction
        proba = float(model.predict_proba(final_input)[0][1])
        action, risk_level = determine_action(proba)

        # Generate explanation
        explanation = generate_human_explanation(proba, txn) if action != Action.ALLOW else "Normal behavior detected"

        # Update metrics
        if not is_simulation and metrics is not None:
            metrics['PREDICTION_COUNT'] += 1
            if action == Action.BLOCK:
                metrics['FRAUD_COUNT'] += 1
                metrics['TOTAL_VALUE_SAVED'] += txn.amount

        # Audit logging
        logging.info(
            f"ID:{request_id} | AMT:{txn.amount} | PROBA:{proba:.4f} | "
            f"ACT:{action.value} | SIM:{is_simulation}"
        )

        # Console logging
        sim_flag = "[SIMULATION] " if is_simulation else ""
        print(
            f"{sim_flag}[{datetime.now().isoformat()}] "
            f"REQ_ID: {request_id} | PROBA: {proba:.4f} | ACTION: {action.value}"
        )

        # Build response
        return {
            "request_id": request_id,
            "status": "success",
            "data": {
                "amount": txn.amount,
                "fraud_probability": round(proba, 4),
                "risk_level": risk_level.value,
                "recommended_action": action.value,
                "explanation": explanation,
                "channel": txn.channel,
                "location": txn.location,
                "sender_bank": txn.sender_bank,
                "bvn_linked": bool(txn.bvn_linked),
                "device_changed": bool(txn.device_changed)
            },
            "meta": {
                "version": "3.0.0",
                "timestamp": datetime.now().isoformat(),
                "model": "sentinel_v3_production"
            },
            "simulation": is_simulation,
            "_internal": {
                "features": final_input.to_dict('records')[0],
                "transaction": txn.model_dump()
            }
        }

    except Exception as e:
        logging.error(f"Processing error for {request_id}: {str(e)}")
        raise

# SHAP EXPLANATION GENERATION
def generate_shap_explanation(
        features_dict: dict,
        feature_names: list,
        explainer
) -> dict:
    """
    Generate SHAP-based feature importance explanation.

    Args:
        features_dict: Dictionary of feature values
        feature_names: List of feature names in correct order
        explainer: SHAP TreeExplainer object

    Returns:
        Dictionary with top fraud/legit drivers
    """
    try:
        # Convert dict to DataFrame
        X_input = pd.DataFrame([features_dict])[feature_names]

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_input)

        if isinstance(shap_values, list):
            impacts = shap_values[1].flatten()
        else:
            impacts = shap_values.flatten()

        feature_importance = dict(zip(feature_names, impacts))
        sorted_importance = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Top features pushing toward fraud
        fraud_drivers = [
            {
                "feature": k.replace('_', ' ').title(),
                "impact": round(float(v), 4),
                "direction": "increases fraud score"
            }
            for k, v in sorted_importance if v > 0.01
        ][:5]

        # Top features pushing toward legit
        legit_drivers = [
            {
                "feature": k.replace('_', ' ').title(),
                "impact": round(abs(float(v)), 4),
                "direction": "decreases fraud score"
            }
            for k, v in sorted_importance if v < -0.01
        ][:5]

        return {
            "top_fraud_drivers": fraud_drivers,
            "top_legitimacy_drivers": legit_drivers,
            "explanation": "Impact values show how much each feature pushed the fraud score up or down"
        }

    except Exception as e:
        logging.error(f"SHAP explanation error: {str(e)}")
        return {
            "error": "Could not generate SHAP explanation",
            "details": str(e)
        }
