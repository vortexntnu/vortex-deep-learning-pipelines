#!/usr/bin/env python3
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from roboflow import Roboflow
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SingleFolderSegmentationDataset
from unet import UNet

# Roboflow parameters
ROBOFLOW_WORKSPACE_NAME = "pipe-92at4"
ROBOFLOW_PROJECT_NAME = "pipeline-detection-2"
ROBOFLOW_PROJECT_VERSION = "5"

# Training hyperparameters
LEARNING_RATE = 1e-5
NUM_EPOCHS = 25

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, masks in dataloaders[phase]:
                inputs = inputs.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f}")

            if phase == "valid" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f"Best validation Loss: {best_loss:.4f}")
    model.load_state_dict(best_model_wts)
    return model


def main():
    # The Roboflow API key is loaded from the environment to avoid hard-coding secrets.
    try:
        roboflow_api_key = os.environ["ROBOFLOW_API_KEY"]
    except KeyError as e:
        raise RuntimeError(
            "ROBOFLOW_API_KEY must be set as an environment variable"
        ) from e
    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE_NAME).project(ROBOFLOW_PROJECT_NAME)
    version = project.version(int(ROBOFLOW_PROJECT_VERSION))
    dataset = version.download("png-mask-semantic")

    train_dir = os.path.join(dataset.location, "train")
    valid_dir = os.path.join(dataset.location, "valid")

    # Transforms
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    mask_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    # Datasets / loaders
    train_dataset = SingleFolderSegmentationDataset(train_dir, transform=transform, mask_transform=mask_transform)
    valid_dataset = SingleFolderSegmentationDataset(valid_dir, transform=transform, mask_transform=mask_transform)

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders = {"train": train_loader, "valid": valid_loader}

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model / loss / optimizer
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    num_epochs = NUM_EPOCHS
    trained_model = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=num_epochs)

    # Save
    torch.save(trained_model.state_dict(), "unet_segmentation.pth")
    print("Model saved as unet_segmentation.pth")


if __name__ == "__main__":
    main()
