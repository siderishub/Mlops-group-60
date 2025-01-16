import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import torchvision.transforms as transforms
from data import data_loader
from visualize import  plot_confusion_matrix
from model import CNN_Baseline
from timm import create_model

def evaluate(model, criterion, test_loader, device, name = "Pretrained"):
    out_dict_test = {
                'name': name,
                'test_acc': [],
                'test_loss': []
                }
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
    out_dict_test['test_acc'].append(test_correct / len(testset))
    out_dict_test['test_loss'].append(np.mean(test_loss))
    print(f"Test Loss: {np.mean(test_loss):.3f}\t Test Accuracy: {out_dict_test['test_acc'][-1] * 100:.1f}%")

    confusion_matrix_path = os.path.join("reports", "figures", "Confusion-Matrix.png")
    plot_confusion_matrix(model, test_loader, device,
                                  class_names=["Pneumonia", "Normal"],
                                  filename=confusion_matrix_path)

    print(f"Model evaluated and results saved: Confusion Matrix -> {confusion_matrix_path}")

    return out_dict_test

if __name__ == "__main__":
    model_name = "Pretrained"  
    print("Using CUDA" if torch.cuda.is_available() else "Using MPS" if torch.backends.mps.is_available() else "Using CPU")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    # Load test data
    batch_size = 64
    testset = data_loader(train=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    model_path = os.path.join("models", f"{model_name}.pt")
    # Load the architecture with pre-trained weights
    model = create_model('mobilenetv3_small_050.lamb_in1k', pretrained=True)
    
    # Adjust the classifier for your task
    model.reset_classifier(num_classes=2)
    
    # Load the saved model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move the model to the device and set to evaluation mode
    model.to(device)
    # Load model dynamically
    # model_path = os.path.join("models", f"{model_name}.pt")
    # model = CNN_Baseline(num_classes=2)
    # model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    # model.to(device)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Evaluate the model
    out_dict_test = evaluate(model=model, criterion=loss_fn, test_loader=test_loader, device=device, name=model_name)

