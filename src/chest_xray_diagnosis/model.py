import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class CNN_Baseline(pl.LightningModule):
    """Convolutional Neural Network base mode."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 32 * 32, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)
    
    def training_step(self, batch):
        """Define training step for Lightning"""
        return None

    #    img, target = batch
    #    y_pred = self(img)
    #    return self.loss_fn(y_pred, target)
    
    def configure_optimizers(self):
        """Define optimizer for Lightning"""
        return None
    #   return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        
if __name__ == "__main__":
    
    #trainer = pl.Trainer(fast_dev_run=True)
    
    cnn = CNN_Baseline()
    print(cnn)