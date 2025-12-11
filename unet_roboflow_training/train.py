#!/usr/bin/env python3
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Import Roboflow and download your dataset.
# Make sure you have installed it via: pip install roboflow
from roboflow import Roboflow

##############################################
# 1. Define the U-Net model
##############################################
class DoubleConv(nn.Module):
    """
    A block with two consecutive convolution layers each followed by
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
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):  # Adjusted for 2 output classes
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

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)  # Adjusted for 2 classes

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
        logits = self.final_conv(c8)  # Raw logits
        return logits

##############################################
# 2. Create a custom Dataset class for segmentation
##############################################
class SingleFolderSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.data_dir = data_dir
        self.image_files = sorted([f for f in os.listdir(data_dir)
                                   if f.endswith('.jpg') and '_mask' not in f])
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.data_dir, image_file)
        mask_path = os.path.join(self.data_dir, image_file.replace('.jpg', '_mask.png'))

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file {mask_path} does not exist for image {image_file}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Load mask as grayscale

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Normalize mask to binary format and convert to long
        mask = (mask > 0.5).long()  # Convert to integer class indices
        mask = mask.squeeze(0)  # Remove the singleton dimension (C=1)

        return image, mask

##############################################
# 3. Define the training loop
##############################################
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'valid']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0

            for inputs, masks in dataloaders[phase]:
                inputs, masks = inputs.to(device), masks.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f}")

            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model

##############################################
# 4. Main function
##############################################
def main():
    rf = Roboflow(api_key="Bc3tBeLXd35djgb8djKN")
    project = rf.workspace("pipe-92at4").project("pipeline-segmentation-nearby")
    version = project.version(2)
    dataset = version.download("png-mask-semantic")

    train_dir = os.path.join(dataset.location, "train")
    valid_dir = os.path.join(dataset.location, "valid")

    transform = transforms.Compose([
        transforms.Resize((544, 960)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Match NVIDIA pipeline
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((544, 960)),
        transforms.ToTensor()
    ])

    train_dataset = SingleFolderSegmentationDataset(train_dir, transform, mask_transform)
    valid_dataset = SingleFolderSegmentationDataset(valid_dir, transform, mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)
    dataloaders = {'train': train_loader, 'valid': valid_loader}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    trained_model = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25)

    torch.save(trained_model.state_dict(), "unet_segmentation.pth")

    trained_model.eval()
    dummy_input = torch.randn(1, 3, 544, 960, device=device)
    torch.onnx.export(
        trained_model,
        dummy_input,
        "unet_segmentation.onnx",
        input_names=["input_1"],
        output_names=["argmax_1"],
        dynamic_axes={"input_1": {0: "batch_size"}, "argmax_1": {0: "batch_size"}},
        opset_version=11
    )
    print("Model exported to unet_segmentation.onnx")

if __name__ == "__main__":
    main()
