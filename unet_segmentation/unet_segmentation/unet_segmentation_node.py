#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image

from unet_segmentation.utils import (
    ResizeIfLargerKeepAspect,
    build_image_transforms,
    load_unet,
    make_overlay,
    mask_to_mono8,
    predict_mask,
    upsample_mask_nearest,
)


def default_config_path() -> str:
    share_dir = Path(get_package_share_directory('unet_segmentation'))
    return str(share_dir / 'config' / 'unet_segmentation.yaml')


class UnetSegmentationNode(Node):
    def __init__(self, config_path: str | None = None):
        super().__init__("unet_segmentation")

        # Optional: ensure a params file exists when started directly
        cfg_path = Path(config_path) if config_path else Path(default_config_path())
        if not cfg_path.is_file():
            self.get_logger().warn(
                f"Params file not found at {cfg_path}. "
                "This is fine if parameters are provided via --params-file in the launch."
            )

        self._declare_and_load_parameters()

        # Resolve device
        self.device = self._make_device(self.device_param)
        self.bridge = CvBridge()

        # Validate model path
        model_p = Path(self.model_path).expanduser()
        if not model_p.exists():
            self.get_logger().fatal(f"Model file not found: {model_p}")
            raise SystemExit(1)

        # Load network
        self.net = load_unet(
            model_path=str(model_p),
            n_classes=self.classes,
            device=self.device,
            bilinear=bool(self.bilinear),
            simple=bool(self.simple),
            logger=self.get_logger(),
        )

        # Build transforms (downscale-only)
        self.image_transforms = build_image_transforms(
            int(self.resize_width), int(self.resize_height)
        )

        # I/O
        qos = QoSProfile(depth=int(self.qos_depth))
        self.subscription = self.create_subscription(
            Image, self.input_topic, self.image_callback, qos
        )
        self.overlay_pub = self.create_publisher(Image, self.overlay_topic, qos)
        self.mask_pub = self.create_publisher(Image, self.mask_topic, qos)

        self.get_logger().info(
            f"Subscribing to '{self.input_topic}', "
            f"publishing overlay to '{self.overlay_topic}' and mask to '{self.mask_topic}'."
        )
        self.get_logger().info(f"Running on device: {self.device}")

    # --- helpers -------------------------------------------------------------

    def _declare_and_load_parameters(self):
        """Declare parameters with defaults and bind them to attributes."""
        defaults = {
            'model_path': 'model/unet.pth',
            'input_topic': 'image_raw',
            'overlay_topic': '/segmentation/overlay',
            'mask_topic': '/segmentation/mask',
            'resize_width': 320,
            'resize_height': 240,
            'keep_original_size': True,
            'mask_threshold': 0.5,
            'bilinear': False,
            'simple': True,
            'classes': 1,  # YAML key is 'classes'
            'device': 'cpu',  # 'cpu', 'cuda', or CUDA index like '0'
            'pred_color': [255, 0, 0],
            'overlay_alpha': 0.4,
            'qos_depth': 3,
        }

        for name, default in defaults.items():
            self.declare_parameter(name, default)

        # Bind as attributes
        self.model_path = self.get_parameter('model_path').value
        self.input_topic = self.get_parameter('input_topic').value
        self.overlay_topic = self.get_parameter('overlay_topic').value
        self.mask_topic = self.get_parameter('mask_topic').value
        self.resize_width = self.get_parameter('resize_width').value
        self.resize_height = self.get_parameter('resize_height').value
        self.keep_original_size = self.get_parameter('keep_original_size').value
        self.mask_threshold = float(self.get_parameter('mask_threshold').value)
        self.bilinear = self.get_parameter('bilinear').value
        self.simple = self.get_parameter('simple').value
        self.classes = int(self.get_parameter('classes').value)
        self.device_param = self.get_parameter('device').value
        self.pred_color = tuple(self.get_parameter('pred_color').value)
        self.overlay_alpha = float(self.get_parameter('overlay_alpha').value)
        self.qos_depth = int(self.get_parameter('qos_depth').value)

    @staticmethod
    def _make_device(device_param: str) -> torch.device:
        if device_param == 'cpu':
            return torch.device('cpu')
        if device_param == 'cuda':
            return torch.device('cuda')
        # allow CUDA index like '0'
        if device_param.isdigit():
            idx = int(device_param)
            return torch.device(f'cuda:{idx}')
        # fallback
        return torch.device(device_param)

    # --- ROS callbacks -------------------------------------------------------

    def image_callback(self, msg: Image):
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            base_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = base_rgb.shape[:2]
            pil_img = PILImage.fromarray(base_rgb)

            # Resize (downscale only) for inference
            resized_pil = ResizeIfLargerKeepAspect(
                int(self.resize_width), int(self.resize_height)
            )(pil_img)
            resized_w, resized_h = resized_pil.size

            image_tensor = self.image_transforms(resized_pil)

            # Predict (in resized space)
            pred_mask_np = predict_mask(
                self.net,
                image_tensor,
                self.device,
                out_threshold=self.mask_threshold,
            )

            # Optionally upsample mask to original size
            if self.keep_original_size:
                mask_out = upsample_mask_nearest(pred_mask_np, orig_w, orig_h)
                base_for_overlay = base_rgb
            else:
                mask_out = pred_mask_np
                base_for_overlay = np.array(resized_pil)

            # Build overlay
            overlay_np = make_overlay(
                base_for_overlay,
                mask_out,
                color=self.pred_color,
                alpha=self.overlay_alpha,
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
