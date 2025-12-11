import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from roboflow import Roboflow
from ultralytics import YOLO

##############################################
# Custom Dataset for Semantic Segmentation
##############################################
class SingleFolderSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        """
        data_dir: directory containing both images and their corresponding masks.
                  Expects image files to be named like 'xxx.jpg' and masks like 'xxx_mask.png'.
        transform: transforms to apply to the image.
        mask_transform: transforms to apply to the mask.
        """
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
        # Derive mask filename (e.g., "image.jpg" -> "image_mask.png")
        mask_path = os.path.join(self.data_dir, image_file.replace('.jpg', '_mask.png'))

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file {mask_path} does not exist for image {image_file}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # load mask as grayscale

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Convert mask to binary values (0 or 1). The ToTensor() produces values in [0,1].
        mask = (mask > 0.5).long()
        # Remove the channel dimension so mask becomes [H, W]
        mask = mask.squeeze(0)
        return image, mask

##############################################
# Custom Training Loop for YOLO Segmentation
##############################################
def custom_train_loop(model, reduce_conv, train_loader, valid_loader, criterion, optimizer, device, num_epochs=50):
    best_loss = float('inf')
    best_model_wts = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        # Set the underlying YOLO model to training mode.
        model.model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)  # masks shape expected: [B, H, W] where H=W=80

            optimizer.zero_grad()
            # Forward pass through the YOLO model.
            outputs = model.model(images)
            # The model may return a nested list/tuple. Unwrap until you get a tensor.
            while isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            # Now outputs has shape [B, C, H, W] (likely [B, 144, 80, 80]).
            # Use the reduction layer to reduce channels from 144 to 1.
            outputs = reduce_conv(outputs)  # now shape [B, 1, 80, 80]

            # Ensure the masks have a channel dimension: convert from [B, 80, 80] to [B, 1, 80, 80].
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            # Compute loss using BCEWithLogitsLoss.
            loss = criterion(outputs, masks.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Training Loss: {epoch_loss:.4f}")

        # (Optional) You could add a validation loop here.

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = model.model.state_dict()

    print("Training complete. Best training loss: {:.4f}".format(best_loss))
    if best_model_wts is not None:
        model.model.load_state_dict(best_model_wts)
    return model

##############################################
# Main function: Download dataset and train
##############################################
def main():
    # -------------------------------
    # Step 1: Download Dataset from Roboflow
    # -------------------------------
    # Replace the API key, workspace, project, and version with your own details.
    rf = Roboflow(api_key="Bc3tBeLXd35djgb8djKN")
    project = rf.workspace("pipe-92at4").project("pipeline-segmentation-nearby")
    version = project.version(2)
    # For semantic segmentation projects, use "png-mask-semantic"
    dataset = version.download("png-mask-semantic")

    # Assume the dataset directory has "train" and "valid" subfolders.
    train_dir = os.path.join(dataset.location, "train")
    valid_dir = os.path.join(dataset.location, "valid")

    # -------------------------------
    # Step 2: Create DataLoaders using the custom dataset
    # -------------------------------
    # Define transforms.
    # We keep the image size at 640×640 (as expected by the model) and downsample masks to 80×80.
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((80, 80), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    train_dataset = SingleFolderSegmentationDataset(train_dir, transform=transform, mask_transform=mask_transform)
    valid_dataset = SingleFolderSegmentationDataset(valid_dir, transform=transform, mask_transform=mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)

    # -------------------------------
    # Step 3: Load the YOLO Segmentation Model
    # -------------------------------
    # You can choose different models such as "yolov8n-seg.pt", "yolov8s-seg.pt", etc.
    model_path = "yolov8n-seg.pt"  # Change as needed.
    model = YOLO(model_path)

    # -------------------------------
    # Step 4: Set up training (device, loss, optimizer, reduction layer)
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.model.to(device)  # Move the underlying YOLO model to the device.

    # Create a 1x1 convolution layer to reduce channel dimension from 144 to 1.
    reduce_conv = nn.Conv2d(144, 1, kernel_size=1).to(device)

    # For binary segmentation, use BCEWithLogitsLoss.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.model.parameters(), lr=1e-4)

    # -------------------------------
    # Step 5: Train the model using the custom training loop
    # -------------------------------
    print("Starting training...")
    trained_model = custom_train_loop(model, reduce_conv, train_loader, valid_loader, criterion, optimizer, device, num_epochs=50)
    print("Training complete.")

    # -------------------------------
    # Step 6: Save the trained model weights
    # -------------------------------
    torch.save(trained_model.model.state_dict(), "yolov8n_seg_trained.pth")
    print("Model saved as yolov8n_seg_trained.pth")

if __name__ == "__main__":
    main()
