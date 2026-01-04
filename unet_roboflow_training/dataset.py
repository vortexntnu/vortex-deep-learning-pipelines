import os
from typing import Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SingleFolderSegmentationDataset(Dataset):
    """
    Expects images (.jpg) and masks (.png) in the same directory:
      frame_0444.jpg
      frame_0444_mask.png
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        mask_transform=None,
    ):
        self.data_dir = data_dir
        self.image_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".jpg") and "_mask" not in f]
        )
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_file = self.image_files[idx]
        image_path = os.path.join(self.data_dir, image_file)

        base, _ = os.path.splitext(image_file)
        mask_file = base + "_mask.png"
        mask_path = os.path.join(self.data_dir, mask_file)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file {mask_path} does not exist for image {image_file}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image) if self.transform else transforms.ToTensor()(image)
        mask = self.mask_transform(mask) if self.mask_transform else transforms.ToTensor()(mask)

        # Binarize (assumes mask pixel values are 0/255 or similar)
        mask = (mask > 0.5).float()
        return image, mask
