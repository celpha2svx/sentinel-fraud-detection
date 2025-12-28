# Sentinel Feature Dictionary (v1.1.0)

Our model uses a combination of raw transaction data and engineered "Behavioral Ratios" to detect fraud.

### Core Features
| Feature | Type | Description |
| :--- | :--- | :--- |
| `amount` | Float | The Naira value of the transaction. |
| `amount_vs_avg_ratio` | Float | How many times larger this txn is compared to the user's normal average. |
| `is_midnight` | Binary | Flagged (1) if the transaction occurs between 12:00 AM and 5:00 AM. |
| `is_ussd` | Binary | Flagged (1) if the USSD channel is used (high-risk in Nigeria). |
| `amount_band` | Categorical | Bucketed transaction size (Micro, Small, Medium, High, Whale). |

### Model Logic
- **Precision:** 99.1% (Near-zero false positives).
- **Recall:** 84.2% (Catches the vast majority of high-value fraud).
- **Explainability:** SHAP values are used to justify every 'BLOCK' action to the compliance team.