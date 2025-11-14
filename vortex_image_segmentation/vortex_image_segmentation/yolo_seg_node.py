"""
ROS2 node for YOLO segmentation: subscribes to images, runs segmentation, and publishes results.
"""

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from ultralytics.engine.results import Results
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from .yolo_seg import YoloSegmentation, YoloSegmentationParams


class YoloSegmentationNode(Node):
    """
    ROS2 node for running YOLO segmentation and publishing results.
    Subscribes to an input image topic, runs segmentation, and publishes output images, masks, and confidences.
    """

    def __init__(self):
        """
        Initialize the YoloSegmentationNode, set up publishers, subscribers, and segmentation model.
        """
        super().__init__("yolo_segmentation_node")
        self.input_topic = "/gripper_camera/image_raw"
        self.output_bbox_topic = "/segmentation/bboxes"
        self.output_mask_topic = "/segmentation/mask"
        self.debug_topic = "/image_debug"
        self.debug = True
        self.imgsz = 640

        self.bridge = CvBridge()
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.create_subscription(
            Image, self.input_topic, self.image_callback, qos_profile
        )
        self.debug_publisher = self.create_publisher(
            Image, self.debug_topic, qos_profile
        )
        self.bbox_pub = self.create_publisher(
            Detection2DArray, self.output_bbox_topic, qos_profile
        )
        self.mask_pub = self.create_publisher(
            Image, self.output_mask_topic, qos_profile
        )

        self.params = self.load_params()
        self.get_logger().info(
            f"Loading YOLO model from: {self.params.model_path}"
        )
        self.segmentation = YoloSegmentation(self.params)
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
        results = self.segmentation.predict(cv_image)
        # When passing in one image to predict, we always want the first result from the list
        result: Results = results[0]

        self.publish_bboxes_and_confidences(result, msg.header)
        self.publish_masks(result, msg.header)

        if self.debug:
            self.publish_debug_image(result, msg.header)

    def publish_bboxes_and_confidences(self, result: Results, header):
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

    def publish_masks(self, result: Results, header):
        """
        Publish segmentation masks as mono8 Image messages.
        """
        masks = result.masks.data.cpu().numpy()
        for mask in masks:
            mask_img = (mask * 255).astype("uint8")
            mask_msg = self.bridge.cv2_to_imgmsg(mask_img, encoding="mono8")
            mask_msg.header = header
            self.mask_pub.publish(mask_msg)

    def publish_debug_image(self, result: Results, header):
        """
        Publish debug visualization image.
        """
        debug_img = self.segmentation.visualize(result)
        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
        debug_msg.header = header
        self.debug_publisher.publish(debug_msg)

    def load_params(self) -> YoloSegmentationParams:
        """
        Load segmentation parameters from ROS2 node parameters.
        Returns:
            YoloSegmentationParams: Segmentation parameters dataclass.
        """
        self.declare_parameter("device", Parameter.Type.STRING)
        self.declare_parameter("model_path", Parameter.Type.STRING)
        self.declare_parameter("confidence_threshold", Parameter.Type.DOUBLE)
        self.declare_parameter("max_detections", Parameter.Type.INTEGER)
        self.declare_parameter("imgsz", Parameter.Type.INTEGER)
        self.declare_parameter("compile", Parameter.Type.BOOL)
        self.declare_parameter("debug", Parameter.Type.BOOL)
        return YoloSegmentationParams(
            device=self.get_parameter("device")
            .get_parameter_value()
            .string_value,
            model_path=self.get_parameter("model_path")
            .get_parameter_value()
            .string_value,
            confidence_threshold=self.get_parameter("confidence_threshold")
            .get_parameter_value()
            .integer_value,
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


def main(args=None):
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
