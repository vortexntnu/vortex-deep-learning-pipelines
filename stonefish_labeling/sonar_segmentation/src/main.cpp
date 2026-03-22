#include "sonar_segmentation/sonar_segmentation_node.hpp"

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<vortex::sonar_segmentation::SonarSegmentationNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
