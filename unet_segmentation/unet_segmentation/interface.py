#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from pathlib import Path
import sys
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from unet_segmentation.config import UnetSegmentationConfig
from unet_segmentation.utils import (
    predict_mask, build_image_transforms, ResizeIfLargerKeepAspect,
    upsample_mask_nearest, mask_to_mono8, make_overlay,
    load_unet,
)

def default_config_path() -> str:
    share_dir = Path(get_package_share_directory('unet_segmentation'))
    return str(share_dir / 'config' / 'unet_segmentation.yaml')

class UnetSegmentationNode(Node):
    def __init__(self, config_path: str | None = None):
        super().__init__("unet_segmentation_node")

        # 1) Resolve config path
        cfg_path = Path(config_path) if config_path else Path(default_config_path())
        if not cfg_path.is_file():
            # Try package share as a fallback if a bad relative path was passed
            pkg_cfg = Path(default_config_path())
            if pkg_cfg.is_file():
                cfg_path = pkg_cfg
            else:
                self.get_logger().fatal(
                    f"Config file not found. Tried: {config_path} and {pkg_cfg}"
                )
                raise SystemExit(1)

        # 2) LOAD the config -> self.cfg  (do this before using self.cfg)
        self.cfg = UnetSegmentationConfig.from_yaml(str(cfg_path))

        # 3) Validate model path AFTER self.cfg exists
        model_p = Path(self.cfg.model_path).expanduser()
        if not model_p.exists():
            self.get_logger().fatal(f"Model file not found: {model_p}")
            raise SystemExit(1)

        self.device = torch.device(self.cfg.device_name)
        self.bridge = CvBridge()

        self.net = load_unet(
            model_path=self.cfg.model_path,
            n_classes=self.cfg.n_classes,
            device=self.device,
            bilinear=self.cfg.bilinear,
            simple=self.cfg.simple,
            logger=self.get_logger(),
        )

        self.image_transforms = build_image_transforms(self.cfg.resize_width, self.cfg.resize_height)
        qos_profile = UnetSegmentationConfig.qos_profile(self.cfg.qos_depth)

        self.subscription = self.create_subscription(Image, self.cfg.input_topic, self.image_callback, qos_profile)
        self.overlay_pub = self.create_publisher(Image, self.cfg.overlay_topic, qos_profile)
        self.mask_pub = self.create_publisher(Image, self.cfg.mask_topic, qos_profile)

        self.get_logger().info(
            f"Subscribing to '{self.cfg.input_topic}', publishing overlay to '{self.cfg.overlay_topic}' "
            f"and mask to '{self.cfg.mask_topic}'."
        )

    def image_callback(self, msg: Image):
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            base_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = base_rgb.shape[:2]
            pil_img = PILImage.fromarray(base_rgb)

            # Resize (downscale only) for inference
            resized_pil = ResizeIfLargerKeepAspect(self.cfg.resize_width, self.cfg.resize_height)(pil_img)
            resized_w, resized_h = resized_pil.size
            image_tensor = self.image_transforms(resized_pil)

            # Predict (in resized space)
            pred_mask = predict_mask(self.net, image_tensor, self.device, out_threshold=self.cfg.mask_threshold)

            # Optionally upsample mask to original size
            if self.cfg.keep_original_size:
                mask_out = upsample_mask_nearest(pred_mask, orig_w, orig_h)
                base_for_overlay = base_rgb  # original size
            else:
                mask_out = pred_mask
                base_for_overlay = np.array(resized_pil)

            # Build overlay at the chosen size
            overlay_np = make_overlay(
                base_for_overlay, mask_out if self.cfg.keep_original_size else pred_mask,
                color=self.cfg.pred_color, alpha=self.cfg.overlay_alpha
            )

            # Publish mask (mono8) and overlay (rgb8)
            mask_mono8 = mask_to_mono8(mask_out)
            mask_msg = self.bridge.cv2_to_imgmsg(mask_mono8, encoding="mono8")
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)

            overlay_msg = self.bridge.cv2_to_imgmsg(overlay_np, encoding="rgb8")
            overlay_msg.header = msg.header
            self.overlay_pub.publish(overlay_msg)

        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')

def main():
    rclpy.init()
    node = UnetSegmentationNode(default_config_path())
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
