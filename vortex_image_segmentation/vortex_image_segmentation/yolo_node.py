
import rclpy
from ultralytics import YOLO
import numpy as np
from PIL import Image as PILImage

from .base_node import BaseSegmentationNode


class YoloSegmentationNode(BaseSegmentationNode):
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame_rgb = cv_image[..., ::-1]
            pil_img = PILImage.fromarray(frame_rgb)
            image_tensor = self.image_transforms(pil_img).unsqueeze(0)  # Add batch dimension
            mask_np, confidence = self.predict(image_tensor)
            if mask_np is None:
                self.get_logger().info('No mask detected; skipping image publish.')
                return
            # If mask is probability, apply threshold
            if hasattr(mask_np, 'dtype') and mask_np.dtype != np.bool_ and hasattr(mask_np, 'max') and mask_np.max() <= 1.0:
                mask_np = (mask_np > self.mask_threshold).astype(np.uint8)
            blended_img = self.blend_image_and_mask(pil_img, mask_np, color=self.mask_color)
            blended_frame_rgb = np.array(blended_img)
            output_msg = self.bridge.cv2_to_imgmsg(blended_frame_rgb, "rgb8")
            output_msg.header = msg.header
            self.publisher.publish(output_msg)
            if confidence is not None:
                self.get_logger().info(f'Confidence score: {confidence}')
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.get_logger().error(f'Failed to process image: {e}\nTraceback:\n{tb}')
    def __init__(self):
        super().__init__('yolo_segmentation_node', mask_color=(0, 255, 0))
        self.model = self.load_model()
        
    def load_model(self):
        self.get_logger().info(f"Model path: {self.model_path}")
        try:
            model = YOLO(self.model_path, task='segment')
            self.get_logger().info(f"Loaded model type: {type(model)}")
            return model
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            return None

    def predict(self, image_tensor):
        results = self.model(image_tensor)
        # Extract mask and confidence from YOLO results
        mask = None
        confidence = None
        if results and hasattr(results[0], 'masks'):
            mask = results[0].masks
        if results and hasattr(results[0], 'probs'):
            confidence = results[0].probs
        return mask, confidence


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
