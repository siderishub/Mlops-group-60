from src.chest_xray_diagnosis.data import data_loader, save_data_as_batches, get_transform
import os


def test_data():
    # Setup
    train_transform = get_transform(train=True, size=128)
    test_transform = get_transform(train=False, size=128)

    # Generate sample data (if not already created)
    if not os.path.exists("data/processed/chest_xray/train") or not os.listdir("data/processed/chest_xray/train"):
        save_data_as_batches(
            train_transform, "data/raw/chest_xray/train", "data/processed/chest_xray/train", overwrite=True
        )
    if not os.path.exists("data/processed/chest_xray/test") or not os.listdir("data/processed/chest_xray/test"):
        save_data_as_batches(
            test_transform, "data/raw/chest_xray/test", "data/processed/chest_xray/test", overwrite=True
        )

    # Test the train dataset
    train_dataset = data_loader(train=True, data_path="data/processed/chest_xray")
    assert len(train_dataset) > 0, "Train dataset is empty"
    for idx in range(len(train_dataset)):
        image, label = train_dataset[idx]
        assert image.shape == (3, 128, 128), f"Train image at index {idx} has invalid shape {image.shape}"
        assert label in [0, 1], f"Train label at index {idx} is invalid: {label}"

    # Test the test dataset
    test_dataset = data_loader(train=False, data_path="data/processed/chest_xray")
    assert len(test_dataset) > 0, "Test dataset is empty"
    for idx in range(len(test_dataset)):
        image, label = test_dataset[idx]
        assert image.shape == (3, 128, 128), f"Test image at index {idx} has invalid shape {image.shape}"
        assert label in [0, 1], f"Test label at index {idx} is invalid: {label}"

    print("All tests passed!")
