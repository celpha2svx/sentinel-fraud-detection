import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
from src.logger import get_logger

logger = get_logger("nigeria_data_generator_v5")


class NigerianDataGenerator:
    def __init__(self, num_users=1000, num_records=50000):
        self.num_users = num_users
        self.num_records = num_records

        self.locations = ['Lagos', 'Abuja', 'PH', 'Ibadan', 'Kano']
        self.legacy_banks = ['GTBank', 'Zenith', 'Access']
        self.fintech_banks = ['Kuda', 'OPay', 'Moniepoint']

        # Approximate of  geo-distance matrix (minutes of travel)
        self.geo_distance = {
            ('Lagos', 'Ibadan'): 120,
            ('Lagos', 'Abuja'): 360,
            ('Lagos', 'PH'): 420,
            ('Lagos', 'Kano'): 720,
            ('Abuja', 'Kano'): 300,
            ('PH', 'Ibadan'): 480
        }

    def _create_users(self):
        users = []
        for i in range(self.num_users):
            bank_type = random.choice(['legacy', 'fintech'])

            avg_amount = (
                random.uniform(2000, 12000)
                if bank_type == 'fintech'
                else random.uniform(5000, 30000)
            )

            users.append({
                'user_id': f"USR_{1000+i}",
                'home_location': random.choice(self.locations),
                'avg_tx_amount': avg_amount,
                'account_balance': random.uniform(80_000, 1_500_000),
                'bank_type': bank_type,
                'sender_bank': random.choice(
                    self.fintech_banks if bank_type == 'fintech' else self.legacy_banks
                ),
                'preferred_channel': (
                    'Mobile App' if bank_type == 'fintech' else random.choice(['POS', 'USSD'])
                ),

                #  STATEFUL RISK FLAGS
                'bvn_linked': 0 if random.random() < 0.15 else 1,
                'device_id': f"DEV_{random.randint(100,999)}",
                'last_location': None,
                'failed_attempts_24h': 0,
                'tx_count_1h': 0,
                'tx_count_24h': 0
            })
        return users

    def generate(self):
        users = self._create_users()
        data = []
        start_date = datetime(2024, 1, 1)

        for i in range(self.num_records):
            user = random.choice(users)
            fraud_reason = "NONE"

            timestamp = start_date + timedelta(minutes=i * 0.8)
            hour = timestamp.hour
            day = timestamp.day

            # Reset rolling counters periodically
            if random.random() < 0.01:
                user['tx_count_1h'] = 0
            if random.random() < 0.005:
                user['tx_count_24h'] = 0
                user['failed_attempts_24h'] = 0

            # BASE BEHAVIOR
            amount = max(
                500,
                np.random.normal(user['avg_tx_amount'], user['avg_tx_amount'] * 0.3)
            )
            channel = user['preferred_channel']
            location = user['home_location']
            device_changed = 0
            is_fraud = 0

            # Salary gravity
            if day in [27, 28, 29, 30, 1, 2]:
                amount *= random.uniform(1.5, 3)

            r = random.random()

            # DEVICE / SIM SWAP
            if r < 0.05:
                device_changed = 1
                fraud_reason = "DEVICE_CHANGE"

            #  PIN PRESSURE
            if r < 0.10:
                user['failed_attempts_24h'] += random.randint(2, 5)

            #  GEO-VELOCITY
            if r < 0.08:
                location = random.choice([l for l in self.locations if l != user['home_location']])

            #  HIGH RISK NIGHT + USSD
            if hour < 5 and channel == 'USSD' and amount > user['avg_tx_amount'] * 8:
                fraud_reason = "MIDNIGHT_USSD_SPIKE"

            #  VELOCITY
            user['tx_count_1h'] += 1
            user['tx_count_24h'] += 1

            #  FRAUD DECISION LOGIC (CAUSAL)
            risk_score = 0

            if user['bvn_linked'] == 0:
                risk_score += 2
            if device_changed:
                risk_score += 2
            if user['failed_attempts_24h'] >= 3:
                risk_score += 2
            if user['tx_count_1h'] >= 5:
                risk_score += 2
            if amount > user['account_balance'] * 0.8:
                risk_score += 2
            if location != user['home_location']:
                risk_score += 1

            if risk_score >= 5:
                is_fraud = 1
                if fraud_reason == "NONE":
                    fraud_reason = "COMBINED_BEHAVIORAL_RISK"

            # Update last known location
            user['last_location'] = location

            data.append({
                'timestamp': timestamp,
                'user_id': user['user_id'],
                'amount': round(amount, 2),
                'user_avg_amount': round(user['avg_tx_amount'], 2),
                'user_balance': round(user['account_balance'], 2),
                'channel': channel,
                'location': location,
                'sender_bank': user['sender_bank'],
                'bvn_linked': user['bvn_linked'],
                'device_changed': device_changed,
                'failed_attempts_24h': user['failed_attempts_24h'],
                'tx_count_1h': user['tx_count_1h'],
                'tx_count_24h': user['tx_count_24h'],
                'fraud_reason': fraud_reason,
                'is_fraud': is_fraud
            })

        df = pd.DataFrame(data)
        logger.info("Nigeria fintech-grade dataset generated.")
        return df



# SAVE
if __name__ == "__main__":
    gen = NigerianDataGenerator(num_users=1532, num_records=75000)
    df = gen.generate()

    target_dir = Path(__file__).resolve().parent.parent
    path = target_dir / "data" / "nigerian_transactions_v5.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)
    logger.info("Saved dataset to data/nigerian_transactions_v5.csv")
