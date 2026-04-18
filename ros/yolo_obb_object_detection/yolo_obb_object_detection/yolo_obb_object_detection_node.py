#!/usr/bin/env python3

import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from yolo_obb_object_detection import yolo_utils


class YoloObbObjectDetection(Node):
    def __init__(self):
        super().__init__('yolo_obb_object_detection')

        self._load_parameters()
        self.model = self._load_model()
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, self.input_topic, self.on_image, qos_profile_sensor_data
        )
        self.pub_dets = self.create_publisher(
            Detection2DArray, self.output_detections_topic, 10
        )
        self.pub_annot = self.create_publisher(
            Image, self.output_annotated_topic, qos_profile_sensor_data
        )

    def _load_parameters(self):
        params = {
            'model_path': Parameter.Type.STRING,
            'confidence_threshold': Parameter.Type.DOUBLE,
            'input_topic': Parameter.Type.STRING,
            'output_detections_topic': Parameter.Type.STRING,
            'output_annotated_topic': Parameter.Type.STRING,
            'device': Parameter.Type.STRING,
        }
        for name, ptype in params.items():
            self.declare_parameter(name, ptype)
            setattr(self, name, self.get_parameter(name).value)

    def _load_model(self):
        mp = os.path.expanduser(self.model_path)
        if not os.path.isabs(mp):
            share = get_package_share_directory('yolo_obb_object_detection')
            mp = os.path.join(share, 'model', mp)
        if not os.path.isfile(mp):
            self.get_logger().error(f"Model not found: {mp}")
            raise FileNotFoundError(mp)
        self.get_logger().info(f"Loading model: {mp}")
        return yolo_utils.load_model(mp, self.confidence_threshold)

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        detections, annot = yolo_utils.process_frame(
            frame, self.model, self.confidence_threshold, self.device
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

        out_msg = self.bridge.cv2_to_imgmsg(annot, encoding='bgr8')
        out_msg.header = msg.header
        self.pub_annot.publish(out_msg)


def main():
    rclpy.init()
    node = YoloObbObjectDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
