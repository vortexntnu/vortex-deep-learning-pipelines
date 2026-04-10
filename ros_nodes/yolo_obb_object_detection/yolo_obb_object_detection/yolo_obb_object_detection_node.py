#!/usr/bin/env python3
import os

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
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

        param_declarations = [
            ("yolo_model", ""),
            ("model_conf", 0.25),
            ("color_image_sub_topic", "/image"),
            ("yolo_detections_pub_topic", "/detections"),
            ("yolo_annotated_pub_topic", "/annotated"),
            ("device", "cpu"),
            ("imgsz", [1984, 1088]),
        ]

        # Declare all parameters in a single, structured call.
        self.declare_parameters("", param_declarations)

        # Map parameter names to attribute names for consistency with the rest of the node.
        param_attr_map = {
            "yolo_model": "model_name",
            "model_conf": "conf",
            "color_image_sub_topic": "image_topic",
            "yolo_detections_pub_topic": "dets_topic",
            "yolo_annotated_pub_topic": "annot_topic",
            "device": "device",
            "imgsz": "imgsz",
        }

        params = self.get_parameters([name for name, _ in param_declarations])
        for param in params:
            attr_name = param_attr_map.get(param.name, param.name)
            setattr(self, attr_name, param.value)
        self.bridge = CvBridge()

        self.model = self._load_model()

        self.sub = self.create_subscription(
            Image, self.image_topic, self.on_image, qos_profile_sensor_data
        )

        self.pub_dets = self.create_publisher(Detection2DArray, self.dets_topic, 10)

        self.pub_annot = self.create_publisher(
            Image, self.annot_topic, qos_profile_sensor_data
        )

    def _load_model(self):
        share = get_package_share_directory("yolo_obb_object_detection")
        default_model = os.path.join(share, "model", self.model_name)

        mp = os.path.expanduser(default_model)

        if not os.path.isabs(mp):
            mp = os.path.join(share, mp)

        if not os.path.isfile(mp):
            self.get_logger().error(f"Model not found: {mp}")
            raise FileNotFoundError(mp)

        self.get_logger().info(f"Loading model: {mp}")
        return yolo_utils.load_model(mp, self.conf)

    @staticmethod
    def _pad_to_stride(img, stride=32):
        h, w = img.shape[:2]
        pad_h = (-h) % stride
        pad_w = (-w) % stride
        left = pad_w // 2
        right = pad_w - left
        return cv2.copyMakeBorder(img, pad_h, 0, left, right, cv2.BORDER_CONSTANT, value=0)

    def on_image(self, msg: Image):
        if msg.encoding in ("mono16", "16UC1"):
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            gray = np.clip(raw // 4, 0, 255).astype(np.uint8)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif msg.encoding in ("mono8", "8UC1"):
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        frame = self._pad_to_stride(frame)

        detections, annot = yolo_utils.process_frame(
            frame, self.model, self.conf, self.device, self.imgsz
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
