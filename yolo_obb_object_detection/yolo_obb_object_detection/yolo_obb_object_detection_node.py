#!/usr/bin/env python3

import os

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
        super().__init__('yolo_obb_object_detection')

        self.get_parameters()

        self.load_model()

        self.sub = self.create_subscription(
            Image, self.color_image_sub_topic, self.on_image, qos_profile_sensor_data
        )
        self.pub_dets = self.create_publisher(
            Detection2DArray, self.yolo_detections_pub_topic, 10
        )
        self.pub_annot = self.create_publisher(
            Image, self.yolo_annotated_pub_topic, qos_profile_sensor_data
        )
        self.bridge = CvBridge()

    def get_parameters(self):
        """Declare parameters with defaults and attach them as class attributes."""
        params = {
            'yolo_model': '_',
            'model_conf': 0.25,
            'color_image_sub_topic': '_',
            'yolo_detections_pub_topic': '_',
            'yolo_annotated_pub_topic': '_',
            'device': '_',
        }

        for name, default in params.items():
            self.declare_parameter(name, default)
            val = self.get_parameter(name).value
            setattr(self, name, val)

    def load_model(self):
        share = get_package_share_directory("yolo_obb_object_detection")

        default_model = os.path.join(share, "model", self.yolo_model)

        mp = os.path.expanduser(default_model)
        if not os.path.isabs(mp):
            mp = os.path.join(share, mp)

        if not os.path.isfile(mp):
            self.get_logger().error(f"Model not found: {mp}")
            raise FileNotFoundError(mp)

        self.model = yolo_utils.load_model(mp, self.model_conf)
        self.conf = self.model_conf


    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        results = self.model.predict(
            source=frame,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )

        r = results[0]
        annot = r.plot()

        det_array = Detection2DArray()
        det_array.header = msg.header

        if r.obb is not None:
            xywhr = r.obb.xywhr.cpu().numpy()
            confs = r.obb.conf.cpu().numpy()
            clss = r.obb.cls.cpu().numpy()

            for (cx, cy, w, h, theta), sc, cid in zip(xywhr, confs, clss):
                det = Detection2D()
                det.header = msg.header

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(int(cid))
                hyp.hypothesis.score = float(sc)
                det.results.append(hyp)

                det.bbox = BoundingBox2D()
                det.bbox.center.position.x = float(cx)
                det.bbox.center.position.y = float(cy)
                det.bbox.center.theta = float(theta)  # radians
                det.bbox.size_x = float(w)
                det.bbox.size_y = float(h)

                det_array.detections.append(det)
        else:
            self.get_logger().warn("No OBB output detected — is this an OBB model?")

        self.pub_dets.publish(det_array)

        out = self.bridge.cv2_to_imgmsg(annot, encoding="bgr8")
        out.header = msg.header
        self.pub_annot.publish(out)


def main():
    rclpy.init()
    node = YoloObjectDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()