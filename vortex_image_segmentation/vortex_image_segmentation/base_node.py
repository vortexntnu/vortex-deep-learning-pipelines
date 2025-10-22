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
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.device_name = self.get_parameter('device').get_parameter_value().string_value
        self.device = torch.device(self.device_name)
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

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, image_tensor):
        """Return mask_np, confidence (or None)"""
        pass

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame_rgb = cv_image[..., ::-1]
            pil_img = PILImage.fromarray(frame_rgb)
            image_tensor = self.image_transforms(pil_img)
            mask_np, confidence = self.predict(image_tensor)
            blended_img = np.array(pil_img)
            blended_img[mask_np == 1] = self.mask_color
            output_msg = self.bridge.cv2_to_imgmsg(blended_img, "rgb8")
            output_msg.header = msg.header
            self.publisher.publish(output_msg)
            if confidence is not None:
                self.get_logger().info(f'Confidence score: {confidence}')
        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')
