#!/usr/bin/env python3

import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8

from yolo_classify import yolo_utils


class ClassifierNode(Node):
    def __init__(self):
        super().__init__('classifier_node')

        self._load_parameters()
        self.model = self._load_model()
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, self.input_topic, self.on_image, qos_profile_sensor_data
        )
        self.pub = self.create_publisher(UInt8, self.output_class_topic, 10)

    def _load_parameters(self):
        params = {
            'model_path': Parameter.Type.STRING,
            'device': Parameter.Type.STRING,
            'input_topic': Parameter.Type.STRING,
            'output_class_topic': Parameter.Type.STRING,
            'imgsz': Parameter.Type.INTEGER,
            'verbose': Parameter.Type.BOOL,
        }
        for name, ptype in params.items():
            self.declare_parameter(name, ptype)
            setattr(self, name, self.get_parameter(name).value)

    def _load_model(self):
        mp = os.path.expanduser(self.model_path)
        if not os.path.isabs(mp):
            share = get_package_share_directory('yolo_classify')
            mp = os.path.join(share, 'model', mp)
        if not os.path.isfile(mp):
            self.get_logger().error(f"Model not found: {mp}")
            raise FileNotFoundError(mp)
        self.get_logger().info(f"Loading model: {mp}")
        return yolo_utils.load_model(mp)

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        class_id, conf, class_name = yolo_utils.process_frame(
            frame, self.model, self.imgsz, self.device, self.verbose
        )

        if self.verbose:
            self.get_logger().info(f"Prediction: {class_name} ({conf:.3f})")

        out = UInt8()
        out.data = class_id
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
