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

app = typer.Typer(help= "CNN")

def train(model, optimizer, criterion, train_loader, trainset, device, num_epochs=10, name="CNN"):
    out_dict = {
        'name': name,
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': []
    }
    
    # Loop through epochs
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()  # Set the model to training mode

        # Initialize epoch-specific metrics
        train_correct = 0
        train_loss = []

        # Loop through mini-batches in the training DataLoader
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc="DataLoader"):
            data, target = data.to(device), target.to(device)  # Move data to the device (GPU/CPU)

            # Zero the gradients computed for each weight
            optimizer.zero_grad()

            # Forward pass: Compute the model's predictions
            output = model(data)

            # Compute the loss
            loss = criterion(output, target)

            # Backward pass: Compute gradients
            loss.backward()

            # Update weights using the optimizer
            optimizer.step()

            # Track the loss for the batch
            train_loss.append(loss.item())

            # Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target == predicted).sum().cpu().item()

        # Calculate training accuracy and loss for the epoch
        out_dict['train_acc'].append(train_correct / len(trainset))
        out_dict['train_loss'].append(np.mean(train_loss))

        # Print training progress for the epoch
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {np.mean(train_loss):.3f} | "
              f"Train Accuracy: {out_dict['train_acc'][-1] * 100:.1f}%")

        # Save the model and visualizations at the last epoch
        if epoch == num_epochs - 1:
            # File and Directory Paths
            model_dir = "models"  # Directory to save the model
            metrics_plot_path = os.path.join("reports", "figures", "metrics.png")  # Path for metrics plot

            # Create directories if they don't exist
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(os.path.dirname(metrics_plot_path), exist_ok=True)

            # Save the model
            model_path = os.path.join(model_dir, f"{name}.pt")
            torch.save(model.state_dict(), model_path)

            # Save the training metrics visualization
            plot_metrics([out_dict], metrics_plot_path)

            print(f"Final model and results saved: Model -> {model_path}, Metrics -> {metrics_plot_path}")

    return out_dict

@app.command()
def main(
    num_epochs: int = typer.Option(10, help="Number of epochs for training"),
    batch_size: int = typer.Option(64, help="Batch size for training"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate for the optimizer."),
    model_name: str = typer.Option("CNN_Baseline", help="Name of the model."),
    device: str = typer.Option(None, help="Device to run training (e.g., 'cuda', 'cpu', 'mps').")
):
    """Main function to configure and train the model."""
    # Select device
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Data transformations
    size = 128
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Data loaders
    trainset = data_loader(train=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Model, optimizer, and loss function
    model = CNN_Baseline(num_classes=2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Train the model
    train(model=model, optimizer=optimizer, criterion=loss_fn, train_loader=train_loader, trainset=trainset, device=device, num_epochs=num_epochs, name=model_name)


if __name__ == "__main__":
    app()