#!/usr/bin/env python3
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from yolo_obb_object_detection import yolo_utils


class YoloObjectDetection(Node):
    def __init__(self):
        super().__init__("yolo_obb_object_detection")

        self.declare_parameter("yolo_model", "")
        self.declare_parameter("model_conf", 0.25)
        self.declare_parameter("color_image_sub_topic", "/image")
        self.declare_parameter("yolo_detections_pub_topic", "/detections")
        self.declare_parameter("yolo_annotated_pub_topic", "/annotated")
        self.declare_parameter("device", "cpu")

        self.model_name = self.get_parameter("yolo_model").value
        self.conf = self.get_parameter("model_conf").value
        self.image_topic = self.get_parameter("color_image_sub_topic").value
        self.dets_topic = self.get_parameter("yolo_detections_pub_topic").value
        self.annot_topic = self.get_parameter("yolo_annotated_pub_topic").value
        self.device = self.get_parameter("device").value

        self.model = self._load_model()

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, self.image_topic, self.on_image, qos_profile_sensor_data
        )

        self.pub_dets = self.create_publisher(
            Detection2DArray, self.dets_topic, 10
        )

        self.pub_annot = self.create_publisher(
            Image, self.annot_topic, qos_profile_sensor_data
        )

    def _load_model(self):
        share = get_package_share_directory("yolo_obb_object_detection")
        model_path = os.path.join(share, "model", self.model_name)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.get_logger().info(f"Loading model: {model_path}")
        return yolo_utils.load_model(model_path, self.conf)

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        detections, annot = yolo_utils.process_frame(
            frame, self.model, self.conf, self.device
        )

        det_array = Detection2DArray()
        det_array.header = msg.header

        for cx, cy, w, h, theta, score, cid in detections:
            det = Detection2D()
            det.header = msg.header

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(cid)
            hyp.hypothesis.score = float(score)
            det.results.append(hyp)

            det.bbox = BoundingBox2D()
            det.bbox.center.position.x = float(cx)
            det.bbox.center.position.y = float(cy)
            det.bbox.center.theta = float(theta)
            det.bbox.size_x = float(w)
            det.bbox.size_y = float(h)

            det_array.detections.append(det)

        self.pub_dets.publish(det_array)

        out_msg = self.bridge.cv2_to_imgmsg(annot, encoding="bgr8")
        out_msg.header = msg.header
        self.pub_annot.publish(out_msg)


def main():
    rclpy.init()
    node = YoloObjectDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
