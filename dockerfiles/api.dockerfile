FROM python:3.11-slim

# Expose the port to match Google Cloud Run's requirements
EXPOSE $PORT

WORKDIR /

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install fastapi uvicorn torch torchvision timm pillow

RUN pip install python-multipart

# Copy application files
COPY src/chest_xray_diagnosis/data.py src/chest_xray_diagnosis/data.py
COPY src/chest_xray_diagnosis/api.py src/chest_xray_diagnosis/api.py
COPY models/Pretrained.pt models/Pretrained.pt

# Set the command to run the FastAPI app
CMD exec uvicorn src.chest_xray_diagnosis.api:app --port $PORT --host 0.0.0.0 --workers 1
