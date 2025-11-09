

import rclpy
from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import List
import numpy as np
from sensor_msgs.msg import Image

from .base_node import BaseSegmentationNode


class YoloSegmentationNode(BaseSegmentationNode):
    def __init__(self):
        super().__init__('yolo_segmentation_node')
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('max_detections', 300)
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.max_detections = self.get_parameter('max_detections').get_parameter_value().integer_value
        self.model = self.load_model()
    
    def image_callback(self, msg: Image) -> None:
        cv_image: np.ndarray = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results: List[Results] = self.model.predict(source=cv_image,
                                                    imgsz=self.imgsz,
                                                    conf=self.confidence_threshold,
                                                    device=self.device,
                                                    max_det=self.max_detections,
                                                    compile=True)
        if results:
            # YOLO always returns a list of Results, one per image. We only pass one image, so use results[0].
            result: Results = results[0]
            annotated_img: np.ndarray = result.plot()
            output_msg: Image = self.bridge.cv2_to_imgmsg(annotated_img, "bgr8")
            output_msg.header = msg.header
            self.publisher.publish(output_msg)
        else:
            self.get_logger().warning('No results to visualize.')

    def load_model(self):
        self.get_logger().info(f"Model path: {self.model_path}")
        try:
            model = YOLO(self.model_path, task='segment')
            self.get_logger().info(f"Loaded model type: {type(model)}")
            return model
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = YoloSegmentationNode()
    node.get_logger().info("Node started.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
