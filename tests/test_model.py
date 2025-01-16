from src.chest_xray_diagnosis.model import CNN_Baseline
from timm import create_model
import torch


def test_model():
    test_CNN_Baseline()
    test_mobilenetv3()


def test_CNN_Baseline():
    model = CNN_Baseline(num_classes=2)  # Initialize model with 2 output classes
    x = torch.randn(1, 3, 128, 128)  # Input tensor with shape (batch_size=1, channels=3, height=128, width=128)
    y = model(x)
    assert y.shape == (1, 2), f"Expected output shape (1, 2), but got {y.shape}"
    print("CNN_Baseline test passed!")


def test_mobilenetv3():
    model = create_model("mobilenetv3_small_050.lamb_in1k", pretrained=True)
    model.reset_classifier(num_classes=2)  # Reset classifier for 2 output classes
    x = torch.randn(1, 3, 128, 128)  # Input tensor with shape (batch_size=1, channels=3, height=128, width=128)
    y = model(x)
    assert y.shape == (1, 2), f"Expected output shape (1, 2), but got {y.shape}"
    print("MobileNetV3 test passed!")
