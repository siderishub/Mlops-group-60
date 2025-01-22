from fastapi.testclient import TestClient
from src.chest_xray_diagnosis.api import app


client = TestClient(app)



def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is up and running!"}