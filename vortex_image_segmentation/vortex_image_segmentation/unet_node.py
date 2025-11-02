import torch
from unet import UNet  # Make sure this import is correct for your project
import numpy as np
from PIL import Image as PILImage

from .base_node import BaseSegmentationNode


class UnetSegmentationNode(BaseSegmentationNode):
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame_rgb = cv_image[..., ::-1]
            pil_img = PILImage.fromarray(frame_rgb)
            image_tensor = self.image_transforms(pil_img).unsqueeze(0)  # Add batch dimension
            mask_np, _ = self.predict(image_tensor)
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
        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')
            
    def __init__(self):
        super().__init__('unet_segmentation_node', mask_color=(255, 0, 0))
        self.net = self.load_model()

    def load_model(self):
        net = UNet(n_channels=3, n_classes=1)
        net.to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device)
        _ = state_dict.pop('mask_values', None)
        net.load_state_dict(state_dict)
        return net

    def predict(self, image_tensor):
        self.net.eval()
        img = image_tensor.unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output = self.net(img).cpu()
            mask = torch.sigmoid(output) > 0.5
        mask_np = mask[0].long().squeeze().numpy()
        return mask_np, None


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = UnetSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
