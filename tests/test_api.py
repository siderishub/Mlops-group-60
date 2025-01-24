from fastapi.testclient import TestClient
from src.chest_xray_diagnosis.api import app
from PIL import Image
import io


client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is up and running!"}


def test_predict_endpoint():
    # Create a dummy image to send to the API
    image = Image.new("RGB", (224, 224), color=(255, 255, 255))  # A white image
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)  # Go to the start of the file-like object

    # Send a POST request with the image
    response = client.post("/predict/", files={"file": ("test_image.jpg", image_bytes, "image/jpeg")})

    # Assert the response
    assert response.status_code == 200
    response_data = response.json()
    assert "predicted_class" in response_data
    assert "probabilities" in response_data
    assert isinstance(response_data["predicted_class"], int)
    assert isinstance(response_data["probabilities"], list)
    assert len(response_data["probabilities"]) > 0
