FROM python:3.9-slim

# setting work dir
WORKDIR /app

# Install system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \ 
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and models
COPY src/ src/
COPY models/ models/
COPY data/ data/
COPY outputs/ outputs
COPY 04_inference.py .

# Define command line
CMD ["python", "04_inference.py"]