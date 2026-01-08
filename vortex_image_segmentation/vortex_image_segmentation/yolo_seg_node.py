"""
ROS2 node for YOLO segmentation: subscribes to images, runs segmentation, and publishes results.
"""

import rclpy
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from ultralytics.engine.results import Results
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from .yolo_seg import YoloSegmentation, YoloSegmentationParams


@dataclass
class YoloNodeParams:
    input_topic: str
    output_bbox_topic: str
    output_mask_topic: str
    debug_topic: str
    debug: bool


class YoloSegmentationNode(Node):
    """
    ROS2 node for running YOLO segmentation and publishing results.
    Subscribes to an input image topic, runs segmentation, and publishes output images, masks, and confidences.
    """

    def __init__(self) -> None:
        """
        Initialize the YoloSegmentationNode, set up publishers, subscribers, and segmentation model.
        """
        super().__init__("yolo_segmentation_node")
        node_params = self.load_node_params()
        self.input_topic: str = node_params.input_topic
        self.output_bbox_topic: str = node_params.output_bbox_topic
        self.output_mask_topic: str = node_params.output_mask_topic
        self.debug_topic: str = node_params.debug_topic
        self.debug: bool = node_params.debug

        self.bridge: CvBridge = CvBridge()
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription: Subscription = self.create_subscription(
            Image, self.input_topic, self.image_callback, qos_profile
        )
        if self.debug:
            self.debug_publisher: Publisher = self.create_publisher(
                Image, self.debug_topic, qos_profile
            )
        self.bbox_pub: Publisher = self.create_publisher(
            Detection2DArray, self.output_bbox_topic, qos_profile
        )
        self.mask_pub: Publisher = self.create_publisher(
            Image, self.output_mask_topic, qos_profile
        )

        self.params: YoloSegmentationParams = self.load_params()
        self.get_logger().info(
            f"Loading YOLO model from: {self.params.model_path}"
        )
        self.segmentation: YoloSegmentation = YoloSegmentation(self.params)
        self.get_logger().info(
            f"Node initialized. Subscribing to '{self.input_topic}'"
        )

    def image_callback(self, msg: Image) -> None:
        """
        Callback for incoming images. Runs segmentation, publishes results and debug images.
        Args:
            msg (sensor_msgs.msg.Image): Input image message.
        """
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results: List[Results] = self.segmentation.predict(cv_image)
        # When passing in one image to predict, we always want the first result from the list
        result: Results = results[0]

        self.publish_bboxes_and_confidences(result, msg.header)
        self.publish_masks(result, msg.header)

        if self.debug:
            self.publish_debug_image(result, msg.header)

    def publish_bboxes_and_confidences(self, result: Results, header: Header) -> None:
        """
        Publish bounding boxes and confidences as Detection2DArray.
        """
        det_array = Detection2DArray()
        det_array.header = header
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, clss):
            det = Detection2D()
            det.header = header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(int(cls_id))
            hyp.hypothesis.score = float(conf)
            det.results.append(hyp)
            bbox = BoundingBox2D()
            bbox.center.position.x = float((x1 + x2) / 2.0)
            bbox.center.position.y = float((y1 + y2) / 2.0)
            bbox.center.theta = 0.0
            bbox.size_x = float(x2 - x1)
            bbox.size_y = float(y2 - y1)
            det.bbox = bbox
            det_array.detections.append(det)
        self.bbox_pub.publish(det_array)

    def publish_masks(self, result: Results, header: Header) -> None:
        """
        Publish segmentation masks as mono8 Image messages.
        """
        if result.masks is None:
            return
        
        masks: np.ndarray = result.masks.data.cpu().numpy()
        for mask in masks:
            mask_img: np.ndarray = (mask * 255).astype("uint8")
            mask_msg: Image = self.bridge.cv2_to_imgmsg(mask_img, encoding="mono8")
            mask_msg.header = header
            self.mask_pub.publish(mask_msg)

    def publish_debug_image(self, result: Results, header: Header) -> None:
        """
        Publish debug visualization image.
        """
        debug_img: np.ndarray = self.segmentation.visualize(result)
        debug_msg: Image = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
        debug_msg.header = header
        self.debug_publisher.publish(debug_msg)

    def load_node_params(self) -> YoloNodeParams:
        """
        Load node-specific parameters (topics, debug).
        Returns:
            YoloNodeParams: Node parameter dataclass.
        """
        self.declare_parameter("input_topic", "/gripper_camera/image_raw")
        self.declare_parameter("output_bbox_topic", "/segmentation/bboxes")
        self.declare_parameter("output_mask_topic", "/segmentation/mask")
        self.declare_parameter("debug_topic", "/image_debug")
        self.declare_parameter("debug", True)
        return YoloNodeParams(
            input_topic=self.get_parameter("input_topic")
            .get_parameter_value()
            .string_value,
            output_bbox_topic=self.get_parameter("output_bbox_topic")
            .get_parameter_value()
            .string_value,
            output_mask_topic=self.get_parameter("output_mask_topic")
            .get_parameter_value()
            .string_value,
            debug_topic=self.get_parameter("debug_topic")
            .get_parameter_value()
            .string_value,
            debug=self.get_parameter("debug")
            .get_parameter_value()
            .bool_value,
        )

    def load_params(self) -> YoloSegmentationParams:
        """
        Load segmentation parameters.
        Returns:
            YoloSegmentationParams: Segmentation parameters dataclass.
        """
        self.declare_parameter("device", Parameter.Type.STRING)
        self.declare_parameter("model_path", Parameter.Type.STRING)
        self.declare_parameter("confidence_threshold", Parameter.Type.DOUBLE)
        self.declare_parameter("max_detections", Parameter.Type.INTEGER)
        self.declare_parameter("imgsz", Parameter.Type.INTEGER)
        self.declare_parameter("compile", Parameter.Type.BOOL)
        return YoloSegmentationParams(
            device=self.get_parameter("device")
            .get_parameter_value()
            .string_value,
            model_path=self.get_parameter("model_path")
            .get_parameter_value()
            .string_value,
            confidence_threshold=self.get_parameter("confidence_threshold")
            .get_parameter_value()
            .double_value,
            max_detections=self.get_parameter("max_detections")
            .get_parameter_value()
            .integer_value,
            imgsz=self.get_parameter("imgsz")
            .get_parameter_value()
            .integer_value,
            compile=self.get_parameter("compile")
            .get_parameter_value()
            .bool_value,
        )


def main(args: Optional[List[str]] = None) -> None:
    """
    Entry point for the ROS2 node. Initializes and spins the YoloSegmentationNode.
    """
    rclpy.init(args=args)
    node = YoloSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
