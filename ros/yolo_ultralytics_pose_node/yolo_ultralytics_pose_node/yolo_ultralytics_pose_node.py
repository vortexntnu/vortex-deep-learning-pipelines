#!/usr/bin/env python3
"""ROS2 node for YOLO-Pose valve orientation detection.

Subscribes to a camera topic, runs YOLO keypoint inference on each frame,
and publishes valve detections with yaw angle derived from two keypoints
(handle-base → handle-tip) rather than OBB geometry.
"""

import math
import os

import cv2

import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


class YoloUltralyticsPoseNode(Node):
    """ROS2 node that runs YOLO-Pose inference to estimate valve handle orientation.

    Publishes a Detection2DArray where bbox.center.theta is the handle yaw
    computed as atan2(kp1.y - kp0.y, kp1.x - kp0.x) over the full [-π, π] range.
    """

    def __init__(self):
        """Initialise node, load parameters and model, register pub/sub."""
        super().__init__('yolo_ultralytics_pose_node')

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
        """Declare and read all ROS2 parameters from the config YAML."""
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
        """Resolve the model path and return a loaded YOLO instance.

        Raises:
            FileNotFoundError: If the .pt weights file cannot be found.
        """
        mp = os.path.expanduser(self.model_path)
        if not os.path.isabs(mp):
            share = get_package_share_directory('yolo_ultralytics_pose_node')
            mp = os.path.join(share, 'model', mp)
        if not os.path.isfile(mp):
            self.get_logger().error(f"Model not found: {mp}")
            raise FileNotFoundError(mp)
        self.get_logger().info(f"Loading model: {mp}")
        return YOLO(mp)

    def on_image(self, msg: Image):
        """Run pose inference on an incoming camera frame and publish results.

        Args:
            msg: Incoming sensor_msgs/Image camera frame.
        """
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model.predict(
            frame, conf=self.confidence_threshold, device=self.device, verbose=False
        )

        det_array = Detection2DArray()
        det_array.header = msg.header

        for r in results:
            boxes = r.boxes
            keypoints = r.keypoints.xy.cpu().numpy()
            xywh = boxes.xywh.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            for i, (cx, cy, w, h) in enumerate(xywh):
                score = float(confs[i])
                cid = int(clss[i])

                kp0_x, kp0_y = keypoints[i][0]  # handle base
                kp1_x, kp1_y = keypoints[i][1]  # handle tip
                theta = math.atan2(kp1_y - kp0_y, kp1_x - kp0_x)

                det = Detection2D()
                det.header = msg.header

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(cid)
                hyp.hypothesis.score = score
                det.results.append(hyp)

                det.bbox = BoundingBox2D()
                det.bbox.center.position.x = float(cx)
                det.bbox.center.position.y = float(cy)
                det.bbox.center.theta = theta
                det.bbox.size_x = float(w)
                det.bbox.size_y = float(h)

                det_array.detections.append(det)

        self.pub_dets.publish(det_array)

        annotated = results[0].plot() if results else frame

        if results:
            for r in results:
                kps = r.keypoints.xy.cpu().numpy()
                boxes = r.boxes.xywh.cpu().numpy()
                for i, kp in enumerate(kps):
                    base = (int(kp[0][0]), int(kp[0][1]))
                    tip = (int(kp[1][0]), int(kp[1][1]))
                    cv2.line(annotated, base, tip, (0, 255, 0), 2)

                    theta_deg = math.degrees(
                        math.atan2(kp[1][1] - kp[0][1], kp[1][0] - kp[0][0])
                    )
                    cx, cy = int(boxes[i][0]), int(boxes[i][1])
                    cv2.putText(
                        annotated,
                        f'{theta_deg:.1f} deg',
                        (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        out_msg.header = msg.header
        self.pub_annot.publish(out_msg)


def main():
    """Entry point: spin the node until shutdown."""
    rclpy.init()
    node = YoloUltralyticsPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
