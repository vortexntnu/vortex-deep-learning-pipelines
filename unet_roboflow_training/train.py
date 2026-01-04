#!/usr/bin/env python3
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# Import Roboflow and download your dataset.
# Make sure you have installed it via: pip install roboflow
from roboflow import Roboflow
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


##############################################
# 1. Define the U-Net model (with a simple UNet)
##############################################
class DoubleConv(nn.Module):
    """A block with two consecutive convolution layers each followed by
    batch normalization and ReLU activation.
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        """For binary segmentation the model outputs 1 channel per pixel.
        """
        super(UNet, self).__init__()
        # Down-sampling path
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Up-sampling path
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Down path
        c1 = self.down1(x)
        p1 = self.pool1(c1)
        c2 = self.down2(p1)
        p2 = self.pool2(c2)
        c3 = self.down3(p2)
        p3 = self.pool3(c3)
        c4 = self.down4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Up path
        u4 = self.up4(bn)
        merge4 = torch.cat([u4, c4], dim=1)
        c5 = self.conv4(merge4)
        u3 = self.up3(c5)
        merge3 = torch.cat([u3, c3], dim=1)
        c6 = self.conv3(merge3)
        u2 = self.up2(c6)
        merge2 = torch.cat([u2, c2], dim=1)
        c7 = self.conv2(merge2)
        u1 = self.up1(c7)
        merge1 = torch.cat([u1, c1], dim=1)
        c8 = self.conv1(merge1)
        output = self.final_conv(c8)
        return output


##############################################
# 2. Create a custom Dataset class for segmentation
#    (Assuming images are .jpg and masks are .png with names like:
#     "frame_0444.jpg" and "frame_0444_mask.png")
##############################################
class SingleFolderSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        """data_dir: directory containing both images and their masks.
        transform: torchvision transforms for the image.
        mask_transform: transforms for the mask.
        """
        self.data_dir = data_dir
        # List only image files (assuming images are .jpg and masks are not included)
        self.image_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith('.jpg') and '_mask' not in f]
        )
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.data_dir, image_file)

        # Derive the mask filename using a fixed .png extension
        base, _ = os.path.splitext(image_file)
        mask_file = base + "_mask.png"
        mask_path = os.path.join(self.data_dir, mask_file)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Mask file {mask_path} does not exist for image {image_file}"
            )

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # load mask as grayscale

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)

        # Binarize the mask (assumes mask pixel values are 0 and 255)
        mask = (mask > 0.5).float()
        return image, mask


##############################################
# 3. Define the training loop
##############################################
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, masks in dataloaders[phase]:
                inputs = inputs.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)
                    # Backward pass and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f}")

            # Deep copy the model if the validation loss improved
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f"Best validation Loss: {best_loss:.4f}")
    model.load_state_dict(best_model_wts)
    return model


##############################################
# 4. Main function: download dataset, create dataloaders, and train
##############################################
def main():
    # ===== Retrieve dataset from Roboflow =====
    # Replace with your actual Roboflow API key, workspace, project, and version.
    rf = Roboflow(api_key="")  # Add your Roboflow API key here
    project = rf.workspace("pipe-92at4").project("pipeline-detection-2")
    version = project.version(5)
    dataset = version.download("png-mask-semantic")

    # With the current dataset, the folder structure is expected as follows:
    # dataset.location/Pipeline-Detection-2-5/train/  --> contains images (.jpg) and masks (.png)
    # dataset.location/Pipeline-Detection-2-5/valid/  --> contains images (.jpg) and masks (.png)
    #
    # Update the directory paths accordingly:
    train_dir = os.path.join(dataset.location, "train")
    valid_dir = os.path.join(dataset.location, "valid")

    # ===== Define transforms =====
    # Resize images and masks to a fixed size (adjust as needed)
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )
    mask_transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    # ===== Create datasets =====
    train_dataset = SingleFolderSegmentationDataset(
        train_dir, transform=transform, mask_transform=mask_transform
    )
    valid_dataset = SingleFolderSegmentationDataset(
        valid_dir, transform=transform, mask_transform=mask_transform
    )

    # ===== Create dataloaders =====
    batch_size = 4  # adjust batch size as needed
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    dataloaders = {'train': train_loader, 'valid': valid_loader}

    # ===== Set device =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ===== Initialize the model =====
    model = UNet(in_channels=3, out_channels=1)
    model = model.to(device)

    # ===== Define loss function and optimizer =====
    # For binary segmentation, BCEWithLogitsLoss is common.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ===== Train the model =====
    num_epochs = 25  # adjust the number of epochs as needed
    trained_model = train_model(
        model, dataloaders, criterion, optimizer, device, num_epochs=num_epochs
    )

    # ===== Save the trained model =====
    torch.save(trained_model.state_dict(), "unet_segmentation.pth")
    print("Model saved as unet_segmentation.pth")


if __name__ == "__main__":
    main()
