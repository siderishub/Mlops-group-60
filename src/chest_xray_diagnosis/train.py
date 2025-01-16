import os
import numpy as np
import pandas as pd
from datetime import datetime
from time import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import CNN_Baseline
from visualize import plot_metrics, plot_confusion_matrix
from data import data_loader
import typer

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import torchvision.transforms as transforms
from tqdm import tqdm
from data import data_loader
from model import CNN_Baseline
from visualize import plot_metrics
from timm import create_model
import typer
from loguru import logger


# Initialize Typer app
app = typer.Typer(help="Train a model")

# Ensure logs folder exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger.add(os.path.join(log_dir, "train.log"), rotation="1 MB", level="INFO", format="{time} {level} {message}")


def train(model, optimizer, criterion, num_epochs=10, name="CNN"):
    """Train the CNN model"""
    out_dict = {
        'name': name,
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': []
    }
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        model.train()
        # For each epoch
        train_correct = 0
        train_loss = []
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc="DataLoader"):
            data, target = data.to(device), target.to(device)
            # Zero the gradients computed for each weight
            optimizer.zero_grad()
            # Forward pass your image through the network
            output = model(data)
            # Compute the loss
            loss = criterion(output, target)
            # Backward pass through the network
            loss.backward()
            # Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            # Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target == predicted).sum().cpu().item()

        out_dict['train_acc'].append(train_correct / len(trainset))
        out_dict['train_loss'].append(np.mean(train_loss))

        logger.info(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {np.mean(train_loss):.3f}, Train Accuracy: {out_dict['train_acc'][-1] * 100:.1f}%")
        # Save the model and visualizations
        if epoch == num_epochs - 1:
            # File and Directory Paths
            model_dir = "models"
            metrics_plot_path = os.path.join("reports", "figures", "metrics.png")
            
            # Create Directories if They Don't Exist
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(os.path.dirname(metrics_plot_path), exist_ok=True)

            # Save Model
            model_path = os.path.join(model_dir, f"{name}.pt")
            torch.save(model.state_dict(), model_path)

            # Save Visualizations
            plot_metrics([out_dict], metrics_plot_path)

            logger.info(f"Model saved at {model_path}")
            logger.info(f"Metrics plot saved at {metrics_plot_path}")
    return out_dict

@app.command()
def main(
    num_epochs: int = typer.Option(2, help="Number of epochs for training."),
    batch_size: int = typer.Option(64, help="Batch size for data loaders."),
    learning_rate: float = typer.Option(1e-3, help="Learning rate for the optimizer."),
    model_name: str = typer.Option("Pretrained", help="Name of the model."),
    device_type: str = typer.Option(None, help="Device to run training (e.g., 'cuda', 'cpu', 'mps')."),
    pretrained: str = typer.Option("True", help="Use a pre-trained model.")
):
    """Main CLI entry point for training the CNN model."""
    logger.info("Starting training script")
    print("Using CUDA" if torch.cuda.is_available() else "Using MPS" if torch.backends.mps.is_available() else "Using CPU")
    global device
    device = torch.device(device_type or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))

    global trainset, train_loader, test_loader
    trainset = data_loader(train=True)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testset = data_loader(train=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Device selected: {device}")
    logger.info(f"Using pretrained model: {pretrained}")

    if pretrained == "True":
        model = create_model('mobilenetv3_small_050.lamb_in1k', pretrained=True)
        in_features = model.get_classifier().in_features
        model.reset_classifier(num_classes=2)
        model.to(device)
        logger.info("Initialized pretrained model")
    else:
        model = CNN_Baseline(num_classes=2).to(device)
        logger.info("Initialized baseline model")


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    logger.info("Starting training")
    train(model=model, optimizer=optimizer, criterion=loss_fn, num_epochs=num_epochs, name=model_name)
    logger.info("Training complete")

if __name__ == "__main__":
    app()