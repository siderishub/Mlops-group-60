import os
import glob
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

class data_loader(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='data/raw/chest_xray'):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpeg')
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y

if __name__ == "__main__":
    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor()          # Convert image to PyTorch Tensor
    ])

    # Create an instance of the data_loader class for training data
    train_dataset = data_loader(train=True, transform=transform, data_path='chest_xray')

    # Check the length of the dataset
    print(f"Number of samples in the training dataset: {len(train_dataset)}")

    # Retrieve and display a sample
    sample_idx = 0
    X, y = train_dataset[sample_idx]
    print(f"Sample {sample_idx} - Image shape: {X.shape}, Label: {y}")

    # Visualize the sample
    plt.imshow(X.permute(1, 2, 0))  # Rearrange dimensions to (H, W, C) for visualization
    plt.title(f"Label: {y}")
    plt.show()

    # Create an instance of the data_loader class for test data
    test_dataset = data_loader(train=False, transform=transform, data_path='chest_xray')

    # Check the length of the test dataset
    print(f"Number of samples in the test dataset: {len(test_dataset)}")
