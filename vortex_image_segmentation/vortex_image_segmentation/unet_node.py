from .base_node import BaseSegmentationNode
import torch
from unet import UNet  # Make sure this import is correct for your project

class UnetSegmentationNode(BaseSegmentationNode):
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
