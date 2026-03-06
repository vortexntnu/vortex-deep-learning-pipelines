#include "camera_segmentation/camera_segmentation.hpp"

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<vortex::camera_segmentation::CameraSegmentationNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
