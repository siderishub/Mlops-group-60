import os
import glob
from PIL import Image
import torch
from torchvision import transforms

class data_loader(torch.utils.data.Dataset):
    def __init__(self, train, data_path='data/processed/chest_xray'):
        """Initialization"""
        data_path = os.path.join(data_path, 'train' if train else 'test')
        self.batch_paths = glob.glob(data_path + '/*.pt')
        self.data = []
        self.labels = []

        for batch_path in self.batch_paths:
            batch = torch.load(batch_path)
            self.data.append(batch['images'])
            self.labels.extend(batch['labels'])

        self.data = torch.cat(self.data)  # Combine all batches into a single tensor
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        """Returns the total number of samples"""
        return len(self.data)

    def __getitem__(self, idx):
        """Generates one sample of data"""
        return self.data[idx], self.labels[idx]

def save_data_as_batches(transform, input_folder, output_folder, batch_size=64, overwrite=False):
    """
    Save processed data in batches as tensors.

    Parameters:
        transform (torchvision.transforms.Compose): Transformations to apply.
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save transformed batches.
        batch_size (int): Number of samples per batch.
        overwrite (bool): Whether to overwrite the output directory.
    """
    if overwrite and os.path.exists(output_folder):
        for root, dirs, files in os.walk(output_folder, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images, labels = [], []
    batch_count = 0

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.jpeg', '.jpg', '.png')):
                input_path = os.path.join(root, file)
                class_name = os.path.split(os.path.split(input_path)[0])[1]
                label = int(class_name == "PNEUMONIA")  # Assuming binary classification

                try:
                    image = Image.open(input_path)
                    transformed_image = transform(image)

                    images.append(transformed_image)
                    labels.append(label)

                    if len(images) == batch_size:
                        batch_path = os.path.join(output_folder, f"batch_{batch_count}.pt")
                        torch.save({'images': torch.stack(images), 'labels': labels}, batch_path)
                        images, labels = [], []
                        batch_count += 1
                        print(f"Saved batch: {batch_path}")
                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")

    # Save remaining data
    if images:
        batch_path = os.path.join(output_folder, f"batch_{batch_count}.pt")
        torch.save({'images': torch.stack(images), 'labels': labels}, batch_path)
        print(f"Saved batch: {batch_path}")

def get_transform(train, size=128):
    """Define transformations for training and testing"""
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


if __name__ == "__main__":
    size = 128
    train_transform = get_transform(True, size)
    test_transform = get_transform(False, size)

    save_data_as_batches(train_transform, 'data/raw/chest_xray/train', 'data/processed/chest_xray/train', overwrite=True)
    save_data_as_batches(test_transform, 'data/raw/chest_xray/test', 'data/processed/chest_xray/test', overwrite=True)
