import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from src.logger import get_logger
from pathlib import Path
logger = get_logger("data_generator")


class NigerianDataGenerator:
    """Generates synthetic Nigerian transaction data with fraud patterns."""

    def __init__(self, num_records=10000):
        self.num_records = num_records
        self.channels = ['USSD', 'Mobile App', 'POS', 'Web']
        self.locations = ['Lagos', 'Abuja', 'Port Harcourt', 'Kano', 'Ibadan', 'Enugu']
        self.banks = ['GTBank', 'Zenith', 'Access', 'Kuda', 'OPay', 'Moniepoint']

    def generate(self):
        logger.info(f"Generating {self.num_records} synthetic Nigerian transactions...")

        data = []
        start_date = datetime(2024, 1, 1)

        for i in range(self.num_records):
            # 1. Temporal Data (Nigerian Time)
            timestamp = start_date + timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            # 2. Amount Patterns (Naira)
            if random.random() < 0.8:
                amount = round(random.uniform(500, 20000), 2)
            else:
                amount = round(random.uniform(50000, 500000), 2)

            # 3. Fraud Logic (The "Injection")
            is_fraud = 0
            # Pattern A: Midnight USSD drain
            if (timestamp.hour >= 0 and timestamp.hour <= 4) and amount > 50000:
                if random.random() < 0.3:
                    is_fraud = 1

            # Pattern B: Unusual Web Activity
            channel = random.choice(self.channels)
            if channel == 'Web' and amount > 100000 and random.random() < 0.15:
                is_fraud = 1

            record = {
                'transaction_id': f"TRX-{1000000 + i}",
                'timestamp': timestamp,
                'sender_bank': random.choice(self.banks),
                'amount': amount,
                'channel': channel,
                'location': random.choice(self.locations),
                'device_type': random.choice(['Android', 'iPhone', 'Feature Phone', 'Web']),
                'is_fraud': is_fraud
            }
            data.append(record)

        df = pd.DataFrame(data)
        logger.info("Generation complete.")
        return df


# To save the data
if __name__ == "__main__":
    gen = NigerianDataGenerator(num_records=50000)
    df = gen.generate()

    # Path resolution
    target_dir = Path(__file__).resolve().parent.parent
    filepath = target_dir / 'data' /  'nigerian_transactions.csv'
    filepath.parent.mkdir(parents=True,exist_ok=True)

    # save data
    df.to_csv(filepath, index=False)
    logger.info("Saved to data/nigerian_transactions.csv")