# Expose the port to match Google Cloud Run's requirements
EXPOSE $PORT

# Set the working directory
WORKDIR /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install fastapi uvicorn torch torchvision timm pillow

# Copy application files
COPY src/api.py /app/api.py
COPY models/Pretrained.pt /app/models/Pretrained.pt

# Set the command to run the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "$PORT"]
