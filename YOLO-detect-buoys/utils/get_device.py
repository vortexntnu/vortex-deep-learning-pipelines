"""This module provides a utility function to get the device to use for training and inference."""

import torch


def get_device() -> torch.device:
    """Get the device to use for training and inference.

    https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu

    Returns:
        torch.device: The device to use for training and inference.
    """
    return torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
