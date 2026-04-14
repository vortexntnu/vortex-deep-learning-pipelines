import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from torchvision import transforms
import cv2
from PIL import Image as PILImage
from ultralytics import YOLO
from rclpy.qos import qos_profile_sensor_data



class ClassifierNode(Node):
    def __init__(self):
        super().__init__('classifier_node')

        # Declare parameter
        self.declare_parameter('model_path', '')

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        if not model_path:
            raise RuntimeError("model_path parameter not set")

        # Load PyTorch model
        
        self.model = YOLO(model_path)
        self.get_logger().info("YOLO classification model loaded")
        self.get_logger().info(f"Loaded model from {model_path}")

        self.bridge = CvBridge()

        
        self.subscription = self.create_subscription(
            Image,
            '/filtered_image',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])


    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

            # Let YOLO handle preprocessing
            results = self.model(
                pil_image,
                imgsz=640,      # IMPORTANT
                verbose=False
            )

            probs = results[0].probs
            class_id = probs.top1
            conf = probs.top1conf
            class_name = self.model.names[class_id]

            self.get_logger().info(
                f"Prediction: {class_name} ({conf:.3f})"
            )

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()