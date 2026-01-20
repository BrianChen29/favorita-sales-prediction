# # For AWS
# FROM public.ecr.aws/lambda/python:3.9

# # Copy requirements.txt
# COPY requirements.txt ${LAMBDA_TASK_ROOT}

# # Install packages
# RUN pip install --no-cache-dir -r requirements.txt

# COPY app.py ${LAMBDA_TASK_ROOT}

# COPY models/ ${LAMBDA_TASK_ROOT}/models/

# COPY src/ ${LAMBDA_TASK_ROOT}/src/

# # Setting execution command line
# CMD [ "app.handler" ]

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
#COPY 04_inference.py .
COPY app.py .

# Define command line
# CMD ["python", "04_inference.py"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]