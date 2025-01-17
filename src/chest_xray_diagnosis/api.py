import os
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms
from timm import create_model
from PIL import Image
import io
from data import get_transform

# Initialize FastAPI app
app = FastAPI()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
#MODEL_PATH = "./models/mobilenetv3.pt"  # Path to your saved model
#model = create_model('mobilenetv3_small_050.lamb_in1k', pretrained=False, num_classes=2)
#model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#model.to(device)
model_name = "Pretrained"
model_path = os.path.join("models", f"{model_name}.pt")
# Load the architecture with pre-trained weights
model = create_model('mobilenetv3_small_050.lamb_in1k', pretrained=True)

# Adjust the classifier for your task
model.reset_classifier(num_classes=2)

# Load the saved model weights
model.load_state_dict(torch.load(model_path, map_location=device))

# Move the model to the device and set to evaluation mode
model.to(device)
model.eval()  # Set the model to evaluation mode

# Define test-time transforms manually
data_config = {
    "input_size": (3, 128, 128),  # Expected input size (channels, height, width)
    "mean": (0.485, 0.456, 0.406),  # Normalization mean
    "std": (0.229, 0.224, 0.225),   # Normalization standard deviation
}
test_transform = get_transform(train=False)

@app.get("/")
def health_check():
    """Health check endpoint to verify API is running."""
    return {"message": "API is up and running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint for image classification using the trained model.
    Input: An image file
    Output: Predicted class and probabilities
    """
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess the image
        image = test_transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Perform inference
        with torch.no_grad():
            outputs = model(image)  # Forward pass
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]  # Convert logits to probabilities
            predicted_class = probabilities.argmax()  # Get the class with the highest probability

        return {
            "predicted_class": int(predicted_class),  # Convert to int for JSON compatibility
            "probabilities": probabilities.tolist()  # Convert numpy array to list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")