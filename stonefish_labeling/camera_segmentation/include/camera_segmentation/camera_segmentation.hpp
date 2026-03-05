#ifndef CAMERA_SEGMENTATION__CAMERA_SEGMENTATION_HPP_
#define CAMERA_SEGMENTATION__CAMERA_SEGMENTATION_HPP_

#include <filesystem>
#include <memory>
#include <unordered_map>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace vortex::camera_segmentation
{

class CameraSegmentationNode : public rclcpp::Node
{
public:
  explicit CameraSegmentationNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  using Image = sensor_msgs::msg::Image;
  using SyncPolicy =
    message_filters::sync_policies::ApproximateTime<Image, Image>;

  void synced_callback(
    const Image::ConstSharedPtr & segmentation_image_color,
    const Image::ConstSharedPtr & front_camera_color);

  message_filters::Subscriber<Image> segmentation_image_color_sub_;
  message_filters::Subscriber<Image> front_camera_color_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  std::filesystem::path out_dir_;

  int frame_counter_{0};
  uint16_t next_id_{0};

  // key = packed RGB, value = dataset class id
  std::unordered_map<uint32_t, uint16_t> rgb_to_id_;
};

}  // namespace vortex::camera_segmentation

#endif // CAMERA_SEGMENTATION__CAMERA_SEGMENTATION_HPP_
