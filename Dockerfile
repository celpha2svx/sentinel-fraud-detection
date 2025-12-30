FROM python:3.11-slim

# 1. Install libgomp1 
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the rest of your code
COPY . .

# 4. Start the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
