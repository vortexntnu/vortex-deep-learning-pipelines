import cv2
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8
from torchvision import transforms
from ultralytics import YOLO


class ClassifierNode(Node):
    def __init__(self):
        super().__init__('classifier_node')

        # Declare parameters
        self.declare_parameter('model_path')
        self.declare_parameter('device')
        self.declare_parameter('image_sub_topic')
        self.declare_parameter('class_pub_topic')

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        if not model_path:
            raise RuntimeError("model_path parameter not set")
        device = self.get_parameter('device').get_parameter_value().string_value
        image_sub_topic = (
            self.get_parameter('image_sub_topic').get_parameter_value().string_value
        )
        class_pub_topic = (
            self.get_parameter('class_pub_topic').get_parameter_value().string_value
        )

        # Load YOLO model
        self.model = YOLO(model_path)
        self.device = device
        self.get_logger().info(
            f"YOLO classification model loaded from {model_path} on device '{device}'"
        )

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image, image_sub_topic, self.image_callback, qos_profile_sensor_data
        )
        self.publisher = self.create_publisher(UInt8, class_pub_topic, 10)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

            # Let YOLO handle preprocessing
            results = self.model(
                pil_image,
                imgsz=640,  # IMPORTANT
                device=self.device,
                verbose=False,
            )

            probs = results[0].probs
            class_id = probs.top1
            conf = probs.top1conf
            class_name = self.model.names[class_id]

            self.get_logger().info(f"Prediction: {class_name} ({conf:.3f})")

            out = UInt8()
            out.data = class_id
            self.publisher.publish(out)

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
