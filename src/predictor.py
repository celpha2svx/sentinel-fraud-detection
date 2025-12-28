import joblib
import pandas as pd
from pathlib import Path
from src.logger import get_logger

logger = get_logger("batch_predictor")


class SentinelPredictor:
    def __init__(self, model_path: str):
        artifact = joblib.load(model_path)
        self.model = artifact['model']
        self.encoders = artifact['encoders']
        self.feature_names = artifact['feature_names']
        logger.info("Sentinel Model and Encoders loaded successfully.")

    def _prepare_features(self, df: pd.DataFrame):
        """Replicates the API logic for bulk data."""
        df = df.copy()

        # 1. Feature Engineering
        df['amount_vs_avg_ratio'] = df['amount'] / df['user_avg_amount']
        df['is_midnight'] = df['hour'].apply(lambda x: 1 if 0 <= x <= 5 else 0)

        bins = [0, 10000, 50000, 200000, 1000000, float('inf')]
        labels = ['Micro', 'Small', 'Medium', 'High', 'Whale']
        df['amount_band'] = pd.cut(df['amount'], bins=bins, labels=labels)

        # 2. Encoding
        for col, le in self.encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # 3. Align Columns
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        return df[self.feature_names]

    def run_inference(self, input_csv: str, output_csv: str):
        logger.info(f"Reading data from {input_csv}...")
        raw_data = pd.read_csv(input_csv)

        # Prepare and Predict
        X = self._prepare_features(raw_data)
        probabilities = self.model.predict_proba(X)[:, 1]

        raw_data['fraud_probability'] = probabilities
        raw_data['action'] = raw_data['fraud_probability'].apply(
            lambda x: 'BLOCK' if x >= 0.8 else ('REVIEW' if x >= 0.5 else 'ALLOW')
        )

        raw_data.to_csv(output_csv, index=False)
        logger.info(f"Batch prediction complete. Results saved to {output_csv}")


if __name__ == "__main__":
    predictor = SentinelPredictor("models/sentinel_v1_final.joblib")
    predictor.run_inference("data/nigerian_transactions.csv", "data/batch_predictions.csv")