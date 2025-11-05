import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image as PILImage
from torchvision import transforms

from .unet import UNet


def predict_mask(
    net: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    out_threshold: float = 0.5,
) -> np.ndarray:
    """Performs inference on a single image tensor.

    Returns a HxW mask (int64) with values in {0,1,...,n_classes-1} for multi-class
    or {0,1} for binary models.
    """
    net.eval()
    img = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        if getattr(net, "n_classes", 1) > 1:
            mask = output.argmax(dim=1)
        else:
            mask = (torch.sigmoid(output) > out_threshold).long()

    return mask[0].long().squeeze().numpy()


def blend_image_and_mask(
    original_image: PILImage.Image,
    mask_array: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.4,
) -> PILImage.Image:
    """Blends a mask over a PIL image using RGBA compositing."""
    original_image = original_image.convert("RGBA")
    overlay = PILImage.new("RGBA", original_image.size, (0, 0, 0, 0))
    overlay_np = np.array(overlay)
    overlay_np[mask_array == 1] = (*color, int(255 * alpha))
    overlay = PILImage.fromarray(overlay_np)
    blended = PILImage.alpha_composite(original_image, overlay)
    return blended.convert("RGB")


class ResizeIfLargerKeepAspect:
    """Downscale a PIL image only if it's larger than the target size, preserving aspect ratio. Never upscales."""

    def __init__(
        self,
        max_width: int,
        max_height: int,
        interpolation=transforms.InterpolationMode.BILINEAR,
    ):
        self.max_width = max_width
        self.max_height = max_height
        self.interpolation = interpolation

    def __call__(self, img: PILImage.Image) -> PILImage.Image:
        w, h = img.size
        if w > self.max_width or h > self.max_height:
            scale = min(self.max_width / w, self.max_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = F.resize(img, (new_h, new_w), interpolation=self.interpolation)
        return img


def build_image_transforms(max_w: int, max_h: int) -> transforms.Compose:
    """Returns a torchvision Compose that resizes (downscale only), converts to tensor, and normalizes."""
    return transforms.Compose(
        [
            ResizeIfLargerKeepAspect(max_width=max_w, max_height=max_h),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_unet(
    model_path: str,
    n_classes: int,
    device: torch.device,
    bilinear: bool,
    simple: bool,
    logger=None,
) -> "UNet":
    # Resolve relative paths to absolute paths
    model_path = os.path.abspath(os.path.expanduser(model_path))

    if logger:
        logger.info(f'Loading model from {model_path}')
        logger.info(f'Using device {device}')
        logger.info(f'simple={simple}, bilinear={bilinear}, n_classes={n_classes}')

    # Create network with named args
    net = UNet(n_channels=3, n_classes=n_classes, simple=simple, bilinear=bilinear)

    try:
        state_dict = torch.load(model_path, map_location=device)
        _ = state_dict.pop('mask_values', None)

        missing, unexpected = net.load_state_dict(state_dict, strict=False)

        if logger:
            if missing:
                logger.warning(f'Missing keys when loading: {missing}')
            if unexpected:
                logger.warning(f'Unexpected keys when loading: {unexpected}')
            logger.info('Model loaded successfully!')
    except FileNotFoundError:
        if logger:
            logger.fatal(
                f"Model file not found at {model_path}. Please check the path."
            )
        raise

    net.to(device)
    net.eval()
    return net


def upsample_mask_nearest(
    mask_np: np.ndarray, target_w: int, target_h: int
) -> np.ndarray:
    """Upsample a HxW mask (int) to (target_h, target_w) via nearest-neighbor."""
    return cv2.resize(
        mask_np.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST
    )


def mask_to_mono8(mask_np: np.ndarray) -> np.ndarray:
    mask_np = mask_np.astype(np.uint8)
    # If it’s binary (only 0/1), scale to 0/255 so it’s visible
    if mask_np.max() <= 1:
        return (mask_np * 255).astype(np.uint8)
    return mask_np


def make_overlay(
    base_rgb_np: np.ndarray, mask_np: np.ndarray, color=(255, 0, 0), alpha=0.4
) -> np.ndarray:
    mask_bin = (mask_np > 0).astype(np.uint8)
    overlay = np.zeros_like(base_rgb_np)
    overlay[mask_bin == 1] = np.array(color, dtype=np.uint8)
    return cv2.addWeighted(base_rgb_np, 1.0, overlay, alpha, 0)
