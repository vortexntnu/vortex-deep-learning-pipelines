#!/usr/bin/env python3

import os
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from rf_deter_detect import rf_detr_utils


class RfDetrObjectDetection(Node):
    def __init__(self):
        super().__init__('rf_deter_detection')
        self._read_parameters()
        self._load_model()

        self.bridge = CvBridge()
        self.image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._processed_frames = 0
        self._processing_seconds = 0.0
        self.sub = self.create_subscription(
            Image,
            self.color_image_sub_topic,
            self.on_image,
            self.image_qos,
        )
        self.pub_dets = self.create_publisher(
            Detection2DArray, self.rf_detr_detections_pub_topic, 10
        )
        self.pub_annot = self.create_publisher(
            Image, self.rf_detr_annotated_pub_topic, self.image_qos
        )

    def _read_parameters(self):
        parameters = {
            'rf_detr_model': 'checkpoint_best_total.pth',
            'model_conf': 0.25,
            'color_image_sub_topic': '/camera/image_raw',
            'rf_detr_detections_pub_topic': '/rf_deter_detection/detections',
            'rf_detr_annotated_pub_topic': '/rf_deter_detection/annotated',
            'device': 'cpu',
            'trust_checkpoint': False,
            'class_names': ['black_pole', 'white_pole'],
            'background_class_id': -1,
            'performance_log_interval': 100,
        }

        for name, default in parameters.items():
            self.declare_parameter(name, default)
            setattr(self, name, self.get_parameter(name).value)

        if not 0.0 <= self.model_conf <= 1.0:
            raise ValueError('model_conf must be between 0.0 and 1.0')

    def _model_path(self) -> str:
        configured_path = os.path.expanduser(self.rf_detr_model)
        if os.path.isabs(configured_path):
            return configured_path

        share = get_package_share_directory('rf_deter_detect')
        return os.path.join(share, 'model', configured_path)

    def _load_model(self):
        model_path = self._model_path()
        if not os.path.isfile(model_path):
            self.get_logger().error(f'Model not found: {model_path}')
            raise FileNotFoundError(model_path)

        self.get_logger().info(
            f'Loading RF-DETR model {model_path} on {self.device}'
        )
        self.model = rf_detr_utils.load_model(
            model_path,
            self.device,
            trust_checkpoint=self.trust_checkpoint,
            class_names=list(self.class_names),
            background_class_id=self.background_class_id,
        )
        session = getattr(self.model, 'session', None)
        if session is not None:
            active_providers = session.get_providers()
            requested_device = rf_detr_utils.normalize_device(self.device)
            if (
                requested_device.startswith('cuda')
                and 'CUDAExecutionProvider' not in active_providers
            ):
                self.get_logger().warning(
                    'ONNX Runtime could not activate CUDA; inference will use CPU. '
                    'Launch with device:=cpu to suppress the CUDA provider warning.'
                )
            self.get_logger().info(
                f"RF-DETR ready with providers: {', '.join(active_providers)}"
            )

    def on_image(self, msg: Image):
        started_at = time.perf_counter()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            detections, annotated = rf_detr_utils.process_frame(
                frame,
                self.model,
                self.model_conf,
                annotate=self.pub_annot.get_subscription_count() > 0,
            )
        except Exception as error:  # Keep the ROS executor alive on a bad frame.
            self.get_logger().error(f'RF-DETR inference failed: {error}')
            return

        det_array = Detection2DArray()
        det_array.header = msg.header

        for result in detections:
            width = result.x2 - result.x1
            height = result.y2 - result.y1

            detection = Detection2D()
            detection.header = msg.header

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(result.class_id)
            hypothesis.hypothesis.score = result.score
            detection.results.append(hypothesis)

            detection.bbox = BoundingBox2D()
            detection.bbox.center.position.x = result.x1 + width / 2.0
            detection.bbox.center.position.y = result.y1 + height / 2.0
            detection.bbox.center.theta = 0.0
            detection.bbox.size_x = width
            detection.bbox.size_y = height
            det_array.detections.append(detection)

        self.pub_dets.publish(det_array)

        if annotated is not None:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            annotated_msg.header = msg.header
            self.pub_annot.publish(annotated_msg)

        self._processed_frames += 1
        self._processing_seconds += time.perf_counter() - started_at
        if (
            self.performance_log_interval > 0
            and self._processed_frames >= self.performance_log_interval
        ):
            average_ms = 1000.0 * self._processing_seconds / self._processed_frames
            self.get_logger().info(
                f'RF-DETR average: {average_ms:.1f} ms/frame '
                f'({1000.0 / average_ms:.1f} FPS)'
            )
            self._processed_frames = 0
            self._processing_seconds = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = RfDetrObjectDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
