import requests
import json

API_URL = "http://localhost:8000/predict"

# Midnight Sweeper
midnight_sweeper = {
    "amount": 450000,
    "user_avg_amount": 15000,
    "user_balance": 500000,
    "channel": "USSD",
    "location": "Abuja",
    "sender_bank": "OPay",
    "bvn_linked": 0,
    "device_changed": 1,
    "failed_attempts_24h": 3,
    "tx_count_1h": 22,
    "tx_count_24h": 45.0,
    "hour": 3,
    "day_of_week": 0,
    "day": 28,
    "is_midnight": 1,
    "is_salary_window": 0,
    "is_ussd": 1,
    "total_spend_24h": 200000.0,
    "amt_to_user_avg_ratio": 30.0,
    "pct_of_balance": 90.0,
    "amt_to_location_avg_ratio": 30.0,
    "channel_velocity_1h": 22.0,
    "pct_balance_withdrawn": 90.0,
    "tx_per_hour": 5.0,
    "is_midnight_high_value": 1
}

response = requests.post(API_URL, json=midnight_sweeper)
print(json.dumps(response.json(), indent=2))