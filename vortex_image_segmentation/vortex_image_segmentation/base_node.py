from abc import ABC, abstractmethod
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from torchvision import transforms
from PIL import Image as PILImage
import numpy as np
import torch


class BaseSegmentationNode(Node, ABC):
    def __init__(self, node_name, mask_color=(0, 255, 0)):
        super().__init__(node_name)
        self.declare_parameter('model_path', '')
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/image_masked')
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.declare_parameter('mask_threshold', 0.5)
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.device_name = self.get_parameter('device').get_parameter_value().string_value
        self.device = torch.device(self.device_name)
        self.mask_threshold = self.get_parameter('mask_threshold').get_parameter_value().double_value
        self.bridge = CvBridge()
        self.mask_color = mask_color
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(Image, self.output_topic, 10)
        self.get_logger().info(f"Node initialized. Subscribing to '{self.input_topic}' and publishing to '{self.output_topic}'.")

    def blend_image_and_mask(self, original_image, mask_array, color, alpha=0.4):
        """
        Blends a mask over an original image.
        """
        original_image = original_image.convert("RGBA")
        overlay = PILImage.new("RGBA", original_image.size, (0, 0, 0, 0))
        overlay_draw = np.array(overlay)
        overlay_draw[mask_array == 1] = (*color, int(255 * alpha))
        overlay = PILImage.fromarray(overlay_draw)
        blended_image = PILImage.alpha_composite(original_image, overlay)
        return blended_image.convert("RGB")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame_rgb = cv_image[..., ::-1]
            pil_img = PILImage.fromarray(frame_rgb)
            image_tensor = self.image_transforms(pil_img)
            mask_np, confidence = self.predict(image_tensor)
            # If mask is probability, apply threshold
            if mask_np.dtype != np.bool_ and mask_np.max() <= 1.0:
                mask_np = (mask_np > self.mask_threshold).astype(np.uint8)
            blended_img = self.blend_image_and_mask(pil_img, mask_np, color=self.mask_color)
            blended_frame_rgb = np.array(blended_img)
            output_msg = self.bridge.cv2_to_imgmsg(blended_frame_rgb, "rgb8")
            output_msg.header = msg.header
            self.publisher.publish(output_msg)
            if confidence is not None:
                self.get_logger().info(f'Confidence score: {confidence}')
        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, image_tensor):
        """Return mask_np, confidence (or None)"""
        pass

