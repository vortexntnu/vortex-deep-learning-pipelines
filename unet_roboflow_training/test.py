#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

##############################################
# 1. Define the U-Net model (same as during training)
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
    def __init__(self, in_channels=3, out_channels=1):
        """
        For binary segmentation the model outputs 1 channel per pixel.
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
# 2. Load the saved model
##############################################
# Set device to CUDA if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and load the saved weights
model = UNet(in_channels=3, out_channels=1)
model_path = "unet_segmentation.pth"  # path to your saved model weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  # set model to evaluation mode

##############################################
# 3. Prepare the test image
##############################################
# Define the transformation (should match the training transform)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Path to the test image (update this path to your test image)
test_image_path = "test.jpg"
image = Image.open(test_image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # add batch dimension
input_tensor = input_tensor.to(device)

##############################################
# 4. Run inference
##############################################
with torch.no_grad():
    output = model(input_tensor)

# Apply sigmoid to convert logits to probabilities and then threshold for binary mask
output_prob = torch.sigmoid(output)
threshold = 0.2
predicted_mask = (output_prob > threshold).float()

# Remove batch and channel dimensions, and convert to NumPy array for visualization
mask_np = predicted_mask.squeeze().cpu().numpy()

##############################################
# 5. Visualize the results
##############################################
plt.figure(figsize=(12, 6))
unique_values = np.unique(mask_np)
print("Unique mask values:", unique_values)

mask_uint8 = (mask_np * 255).astype("uint8")

plt.imsave("predicted_mask.png", mask_uint8, cmap="gray")
print("Saved predicted mask to predicted_mask.png")

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# Display the predicted mask
plt.subplot(1, 2, 2)
plt.imshow(mask_uint8, cmap='gray')
plt.title("Predicted Mask")
plt.axis("off")

plt.savefig("test_output.png")
print("Output saved to test_output.png")

