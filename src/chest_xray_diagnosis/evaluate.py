import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from data import data_loader
from visualize import plot_confusion_matrix
from model import CNN_Baseline
from timm import create_model
import typer
from loguru import logger
from datetime import datetime

app = typer.Typer(help="Evaluate a model")
# Get current date for logging and file naming
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

log_dir = "logs"
evaluate_log_dir = os.path.join(log_dir, "evaluate_logs")
os.makedirs(evaluate_log_dir, exist_ok=True)

log_file = os.path.join(evaluate_log_dir, f"evaluate_log_{current_date}.log")
logger.add(log_file, rotation="1 MB", level="INFO", format="{time} {level} {message}")


def evaluate(model, criterion, test_loader, device, name="Pretrained"):
    out_dict_test = {"name": name, "test_acc": [], "test_loss": []}
    test_loss = []
    test_correct = 0
    model.eval()
    logger.info("Starting evaluation")

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        test_loss.append(criterion(output, target).cpu().item())
        predicted = output.argmax(1)
        test_correct += (target == predicted).sum().cpu().item()
    out_dict_test["test_acc"].append(test_correct / len(testset))
    out_dict_test["test_loss"].append(np.mean(test_loss))
    logger.info(f"Test Loss: {np.mean(test_loss):.3f}	 Test Accuracy: {out_dict_test['test_acc'][-1] * 100:.1f}%")
    confusion_matrix_path = os.path.join("reports", "figures", f"Confusion-Matrix_{current_date}.png")
    plot_confusion_matrix(
        model, test_loader, device, class_names=["Pneumonia", "Normal"], filename=confusion_matrix_path
    )

    logger.info(f"Confusion matrix saved at {confusion_matrix_path}")

    return out_dict_test


@app.command()
def main(
    model_name: str = typer.Option("Pretrained", help="Name of the model to evaluate."),
    batch_size: int = typer.Option(64, help="Batch size for data loaders."),
    device_type: str = typer.Option(None, help="Device to run evaluation (e.g., 'cuda', 'cpu', 'mps')."),
    pretrained: str = typer.Option("True", help="Use a pretrained model (True) or baseline model (False)."),
):
    """
    Main CLI entry point for evaluating the CNN model.
    """
    logger.info("Starting evaluation script")

    print(
        "Using CUDA" if torch.cuda.is_available() else "Using MPS" if torch.backends.mps.is_available() else "Using CPU"
    )
    device = torch.device(
        device_type or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )

    # Load test data
    global testset
    testset = data_loader(train=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Device selected: {device}")
    logger.info(f"Using pretrained model: {pretrained}")
    # Load the model
    model_path = os.path.join("models", f"{model_name}.pt")
    if pretrained == "True":
        model = create_model("mobilenetv3_small_050.lamb_in1k", pretrained=True)
        model.reset_classifier(num_classes=2)
        logger.info("Initialized pretrained model")
    else:
        model = CNN_Baseline(num_classes=2)
        logger.info("Initialized baseline model")

    # Load model weights
    if not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found at {model_path}")
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Loaded model weights from {model_path}")

    # Move the model to the device
    model.to(device)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Evaluate the model
    logger.info("Starting evaluation process")
    evaluate(model=model, criterion=loss_fn, test_loader=test_loader, device=device, name=model_name)
    logger.info("Evaluation complete")


if __name__ == "__main__":
    app()
