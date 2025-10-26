from dataclasses import dataclass

import torch
import yaml
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy


@dataclass
class UnetSegmentationConfig:
    model_path: str
    input_topic: str
    overlay_topic: str
    mask_topic: str
    resize_width: int
    resize_height: int
    keep_original_size: bool
    mask_threshold: float
    bilinear: bool
    simple: bool
    n_classes: int
    device_name: str
    pred_color: tuple[int, int, int]
    overlay_alpha: float
    qos_depth: int

    @staticmethod
    def from_yaml(path: str) -> "UnetSegmentationConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        p = data.get("unet_segmentation_node", {}).get("ros__parameters", {})
        device_name = p.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        return UnetSegmentationConfig(
            model_path=p["model_path"],
            input_topic=p.get("input_topic", "/image_color"),
            overlay_topic=p.get("overlay_topic", "/segmentation/overlay"),
            mask_topic=p.get("mask_topic", "/segmentation/mask"),
            resize_width=p.get("resize_width", 320),
            resize_height=p.get("resize_height", 240),
            keep_original_size=p.get("keep_original_size", True),
            mask_threshold=p.get("mask_threshold", 0.5),
            bilinear=p.get("bilinear", False),
            simple=p.get("simple", True),
            n_classes=p.get("classes", 1),
            device_name=device_name,
            pred_color=tuple(p.get("pred_color", [255, 0, 0])),
            overlay_alpha=p.get("overlay_alpha", 0.4),
            qos_depth=p.get("qos_depth", 3),
        )

    @staticmethod
    def qos_profile(depth: int = 3) -> QoSProfile:
        return QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
        )
