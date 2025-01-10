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
from model import CNN_Simple

# Your custom functions for plotting
from visualize import plot_metrics, plot_confusion_matrix
from data import data_loader

def train(model, optimizer, criterion, num_epochs=10, name="CNN", folder=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")):
    out_dict = {
                'name': name,
                'train_acc': [],
                'test_acc': [],
                'train_loss': [],
                'test_loss': []
                }
    run_start = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    for epoch in tqdm(range(num_epochs), unit='epoch'):
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

        # Compute the test accuracy
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(criterion(output, target).cpu().item())
            predicted = output.argmax(1)
            test_correct += (target == predicted).sum().cpu().item()
        out_dict['train_acc'].append(train_correct / len(trainset))
        out_dict['test_acc'].append(test_correct / len(testset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1] * 100:.1f}%\t test: {out_dict['test_acc'][-1] * 100:.1f}%")
        os.makedirs(run_start, exist_ok=True)
        filename_model = os.path.join(run_start, f"{name}-model", f"{epoch}.pt")    
        filename_plot_metrics = os.path.join(run_start, f"metrics.png")
        filename_plot_conf_mat = os.path.join(run_start, f"Confusion-Matrix", f"{epoch}.png")
        df=pd.DataFrame(out_dict)
        df.to_csv(os.path.join(run_start, f'training_metrics{epoch}.csv'), index=False)

        os.makedirs(os.path.dirname(filename_model), exist_ok=True)
        os.makedirs(os.path.dirname(filename_plot_conf_mat), exist_ok=True)
        os.makedirs(os.path.dirname(filename_plot_metrics), exist_ok=True)
        t = time()
        model.save(filename_model)
        plot_metrics([out_dict], filename_plot_metrics)
        plot_confusion_matrix(model, test_loader, device,
                              class_names=["Pneumonia", "Normal"],
                              filename=filename_plot_conf_mat)

        print(f"Took {time()-t:.2f} seconds to save and plot results")

        

    return out_dict


if __name__ == "__main__":
    print("Using CUDA" if torch.cuda.is_available() else "Using MPS" if torch.backends.mps.is_available() else "Using CPU")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


    size = 128
    train_transform = train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.Resize((size, size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
                                    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
                                    transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64
    trainset = data_loader(train=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testset = data_loader(train=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


    lr = 1e-3


    model = CNN_Simple(num_classes=2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()


    out_dict = train(model=model, optimizer=optimizer, criterion=loss_fn, num_epochs=2)