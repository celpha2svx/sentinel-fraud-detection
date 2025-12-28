import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
from src.logger import get_logger

logger = get_logger("data_generator_v3")


class NigerianDataGenerator:
    def __init__(self, num_users=1000, num_records=50000):
        self.num_users = num_users
        self.num_records = num_records

        self.banks = ['GTBank', 'Zenith', 'Access', 'Kuda', 'OPay', 'Moniepoint']
        self.channels = ['USSD', 'Mobile App', 'POS', 'Web']
        self.locations = ['Lagos', 'Abuja', 'PH', 'Kano', 'Ibadan']

    def _create_users(self):
        users = []
        for i in range(self.num_users):
            users.append({
                'user_id': f"USR_{1000 + i}",
                'home_location': random.choice(self.locations),
                'avg_tx_amount': random.uniform(2000, 15000),
                'preferred_channel': random.choice(['Mobile App', 'POS', 'USSD']),
                'account_balance': random.uniform(50000, 1_000_000)
            })
        return users

    def generate(self):
        logger.info("Creating user behavioral profiles...")
        users = self._create_users()

        data = []
        start_date = datetime(2024, 1, 1)

        logger.info(f"Generating {self.num_records} transactions with realistic noise...")
        for i in range(self.num_records):
            user = random.choice(users)

            # Base timestamp
            timestamp = start_date + timedelta(
                seconds=i * (30 * 60 / self.num_users)
            )

            # ---------------- NORMAL BEHAVIOR ----------------
            is_fraud = 0
            amount = np.random.normal(
                user['avg_tx_amount'],
                user['avg_tx_amount'] * 0.25
            )
            amount = max(500, round(amount, 2))

            channel = user['preferred_channel']
            location = user['home_location']

            # ---------------- NOISE & FRAUD INJECTION ----------------
            rand_val = random.random()

            # Tricky Fraud (Looks Legit)
            if rand_val < 0.01:
                is_fraud = 1
                amount = user['avg_tx_amount'] * random.uniform(0.8, 1.2)
                channel = user['preferred_channel']
                location = user['home_location']

            # Suspicious Legit (False Positives)
            elif rand_val < 0.05:
                is_fraud = 0
                amount = random.uniform(80_000, 200_000)
                channel = 'USSD'
                timestamp = timestamp.replace(hour=random.randint(0, 3))

            # Obvious Fraud (Classic)
            elif rand_val < 0.07:
                is_fraud = 1
                amount = random.uniform(250_000, 500_000)
                channel = 'USSD'
                location = random.choice(
                    [l for l in self.locations if l != user['home_location']]
                )
                timestamp = timestamp.replace(hour=random.randint(0, 4))

            # Rapid Fire Drain (Velocity Fraud)
            elif rand_val < 0.09:
                is_fraud = 1
                amount = user['account_balance'] * random.uniform(0.6, 0.85)
                channel = random.choice(['Web', 'USSD'])

            # Channel Drift Fraud (Fraud hiding in safe channels)
            elif rand_val < 0.11:
                is_fraud = 1
                amount = random.uniform(40_000, 90_000)
                channel = 'Mobile App'
                location = user['home_location']

            # -------------------------------------------------
            data.append({
                'timestamp': timestamp,
                'user_id': user['user_id'],
                'amount': round(amount, 2),
                'user_avg_amount': user['avg_tx_amount'],
                'user_balance': user['account_balance'],
                'channel': channel,
                'location': location,
                'sender_bank': random.choice(self.banks),
                'is_fraud': is_fraud
            })

        df = pd.DataFrame(data)
        logger.info("Data generation complete.")
        return df


# ---------------- SAVE DATA ----------------
if __name__ == "__main__":
    gen = NigerianDataGenerator(
        num_users=1000,
        num_records=55200
    )

    df = gen.generate()

    target_dir = Path(__file__).resolve().parent.parent
    filepath = target_dir / 'data' / 'nigerian_transactions.csv'
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filepath, index=False)
    logger.info("Saved dataset to data/nigerian_transactions.csv")
