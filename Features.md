# 📊 Sentinel Feature Engineering: The Behavioral Science

Sentinel doesn't just look at transaction amounts; it analyzes the **intent** and **behavior** behind the money. We have engineered 26+ specific features designed to detect Nigerian fraud archetypes like "Account Takeovers," "Midnight Sweepers," and "USSD Velocity Bots."

---

## 🧠 Core Feature Logic

### 1. 🕒 Temporal & Behavioral Rhythms
These features detect shifts in a user's biological clock and spending habits.
* **`is_midnight`**: Flags transactions between 12:00 AM and 5:00 AM. While not illegal, high-value transfers during these hours are the primary window for "Midnight Sweepers."
* **`is_salary_window`**: Identifies transactions occurring between the 25th and 5th of the month. This helps the model distinguish between a legitimate "Salary Splurge" and fraud.
* **`hour` & `day_of_week`**: Captures the rhythmic signature of the user (e.g., does this user usually spend on Sunday afternoons?).

### 2. 📈 Velocity & Frequency (The "Bot" Detectors)
Fraudsters often try to move money quickly before a card is blocked.
* **`tx_count_1h` & `tx_count_24h`**: Detects rapid-fire transactions that suggest automated scripts or high-pressure "Yahoo-Yahoo" withdrawals.
* **`channel_velocity_1h`**: Measures how many times a specific channel (like USSD) is used in an hour. A spike here often indicates a compromised PIN.
* **`tx_per_hour`**: The baseline frequency of the user's digital heartbeat.

### 3. ⚖️ Financial Ratios (The "Surgical" Features)
These are our strongest predictors, looking at deviations from the user's financial "Normal."
* **`amt_to_user_avg_ratio`**: If a user typically spends ₦5,000 and suddenly spends ₦450,000, this ratio hits **90.0x**, triggering an immediate "High Risk" flag.
* **`pct_of_balance`**: A fraudster's goal is to drain the account. Legitimate users rarely transfer 95%+ of their total balance in a single go.
* **`pct_balance_withdrawn`**: Specifically tracks how much of the available liquidity is being moved out of the ecosystem.

### 4. 📍 Geolocation & Infrastructure
* **`location_avg_30d`**: The "Economic Signature" of a location. Abuja spending patterns differ significantly from rural village patterns.
* **`amt_to_location_avg_ratio`**: Detects "Outlier Spending." If a transaction in a low-spend area is massive, it suggests a stolen device being used in a remote location.
* **`device_changed`**: The ultimate red flag. Combined with a high-value transfer, this is a 99% indicator of an Account Takeover (ATO).

---

## 🏗️ Technical Implementation

### Data Pipeline
1.  **Imputation**: `ROLLING_COLS` (like 24h spend) are filled with `0` rather than means to preserve the "Quiet Account" signal.
2.  **Encoding**: Categorical features (`channel`, `location`, `sender_bank`) are processed via `LabelEncoder` to maintain a light memory footprint for the API.
3.  **Scaling**: We use a **Random Forest/XGBoost Ensemble**, which is robust to feature scales, allowing us to keep the raw Naira values for better explainability.

### The "Naira Impact" Calculation
We use these features to calculate the **Net Business Value**:
$$Net\ Value = (Fraud\ Caught \times Amount) - (False\ Positives \times ₦2,500)$$
* *Note: ₦2,500 represents the estimated cost of customer service friction and reputation damage per false alarm.*

---

## 🛡️ Top 5 Fraud Drivers (SHAP Importance)
According to our SHAP analysis, the model relies most heavily on:
1.  **`amt_to_user_avg_ratio`** (Deviation from habit)
2.  **`device_changed`** (Hardware security)
3.  **`is_midnight_high_value`** (Timing risk)
4.  **`pct_balance_withdrawn`** (Account draining)
5.  **`failed_attempts_24h`** (Brute force signal)