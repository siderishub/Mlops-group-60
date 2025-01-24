import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from datetime import datetime


def plot_metrics(out_dicts, filename=""):
    epochs = range(1, len(out_dicts[0]["train_acc"]) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot accuracies
    for out_dict, col in zip(out_dicts, ["blue", "orange"]):
        ax1.plot(
            epochs,
            [x * 100 for x in out_dict["train_acc"]],
            label=f'{out_dict["name"]} Train Accuracy',
            marker="o",
            color=col,
            linestyle="solid",
        )
        # ax1.plot(epochs, [x*100 for x in out_dict['test_acc']], label=f'{out_dict["name"]} Test Accuracy', marker='o', color=col)
    ax1.set_title("Accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy [%]")
    ax1.set_xticks(epochs)
    ax1.legend()
    ax1.grid(True)

    # Plot losses
    for out_dict, col in zip(out_dicts, ["blue", "orange"]):
        ax2.plot(
            epochs,
            out_dict["train_loss"],
            label=f'{out_dict["name"]} Train Loss',
            marker="o",
            color=col,
            linestyle="solid",
        )
        # ax2.plot(epochs, out_dict['test_loss'], label=f'{out_dict["name"]} Test Loss', marker='x', color=col)
    ax2.set_title("Loss over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_xticks(epochs)
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # Save plot if specified
    if filename:
        current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = filename.replace(".png", f"_{current_date}.png")
        plt.savefig(filename)
    plt.close()


def plot_confusion_matrix(model, test_loader, device, class_names, filename=""):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if filename:
        current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = filename.replace(".png", f"_{current_date}.png")
        plt.savefig(filename)
    plt.close()
