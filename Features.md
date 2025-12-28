# Sentinel Feature Dictionary (v1.1.0)

| Feature | Description | Type |
| :--- | :--- | :--- |
| `amount` | Current transaction value in Naira | Float |
| `amount_vs_avg_ratio` | `amount` / `user_avg_amount` (The "Shock" indicator) | Float |
| `is_midnight` | 1 if transaction is between 00:00 - 05:00 | Binary |
| `is_ussd` | 1 if channel is USSD | Binary |
| `user_balance` | Account balance at time of transaction | Float |
| `tx_count_24h` | Number of transactions in last 24h | Integer |

**Model Thresholds:**
- **BLOCK:** > 0.80 probability
- **REVIEW (2FA):** 0.50 - 0.79 probability
- **ALLOW:** < 0.50 probability