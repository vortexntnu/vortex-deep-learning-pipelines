#include "camera_segmentation/camera_segmentation.hpp"
#include "camera_segmentation/dataset_utils.hpp"

#include <limits>

namespace vortex::camera_segmentation {

CameraSegmentationNode::CameraSegmentationNode(
    const rclcpp::NodeOptions &options)
    : Node("camera_segmentation_node", options) {
  const auto segmentation_image_color_sub_topic =
      this->declare_parameter<std::string>("segmentation_image_sub_topic");

  const auto front_camera_color_sub_topic =
      this->declare_parameter<std::string>("color_image_sub_topic");

  const auto sync_tolerance_ms = this->declare_parameter<int>("sync_tolerance_ms");

  std::string default_out = []() {
    const char *home = std::getenv("HOME");
    if (home && *home) {
      return std::string(home) + "/seg_frames";
    }
    return std::string("/tmp/seg_frames");
  }();

  out_dir_ = this->declare_parameter<std::string>("output_dir", default_out);

  dataset_utils::prepare_output_directory(out_dir_, get_logger());

  frame_counter_ = 0;
  next_id_ = 0;
  rgb_to_id_.clear();

  auto qos = rclcpp::SensorDataQoS();
  segmentation_image_color_sub_.subscribe(
      this, segmentation_image_color_sub_topic, qos.get_rmw_qos_profile());
  front_camera_color_sub_.subscribe(this, front_camera_color_sub_topic,
                                    qos.get_rmw_qos_profile());

  auto sync_tolerance = std::chrono::milliseconds{sync_tolerance_ms};

  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
    SyncPolicy(10), segmentation_image_color_sub_,
    front_camera_color_sub_);

  sync_->setMaxIntervalDuration(sync_tolerance);

  sync_->registerCallback(std::bind(&CameraSegmentationNode::synced_callback,
                                    this, std::placeholders::_1,
                                    std::placeholders::_2));
}

void CameraSegmentationNode::synced_callback(
    const Image::ConstSharedPtr &segmentation_image_color,
    const Image::ConstSharedPtr &front_camera_color) {
  cv::Mat seg_color;
  cv::Mat front_color;

  try {
    seg_color = cv_bridge::toCvShare(segmentation_image_color, "bgr8")->image;
    front_color = cv_bridge::toCvShare(front_camera_color, "bgr8")->image;
  } catch (const std::exception &e) {
    RCLCPP_WARN(get_logger(), "cv_bridge conversion failed: %s", e.what());
    return;
  }

  if (seg_color.empty() || front_color.empty()) {
    RCLCPP_WARN(get_logger(), "Received empty image.");
    return;
  }

  if (seg_color.size() != front_color.size()) {
    RCLCPP_WARN(get_logger(),
                "Segmentation image and front image sizes do not match: "
                "seg=%dx%d front=%dx%d",
                seg_color.cols, seg_color.rows, front_color.cols,
                front_color.rows);
    return;
  }

  cv::Mat id_mask(seg_color.rows, seg_color.cols, CV_16UC1);

  for (int y = 0; y < seg_color.rows; ++y) {
    const cv::Vec3b *seg_p = seg_color.ptr<cv::Vec3b>(y);
    uint16_t *id_p = id_mask.ptr<uint16_t>(y);

    for (int x = 0; x < seg_color.cols; ++x) {
      const uint32_t rgb_key = dataset_utils::pack_rgb(seg_p[x]);

      auto it = rgb_to_id_.find(rgb_key);
      if (it == rgb_to_id_.end()) {
        if (next_id_ == std::numeric_limits<uint16_t>::max()) {
          RCLCPP_ERROR(get_logger(), "Too many unique RGB values for uint16 "
                                     "IDs. Stopping frame save.");
          return;
        }

        const uint16_t new_id = next_id_;
        ++next_id_;
        it = rgb_to_id_.emplace(rgb_key, new_id).first;
      }

      id_p[x] = it->second;
    }
  }

  std::string image_path;
  std::string mask_path;

  if (!dataset_utils::save_frame_pair(out_dir_, frame_counter_, front_color,
                                      id_mask, get_logger(), &image_path,
                                      &mask_path)) {
    return;
  }

  dataset_utils::save_legend_csv(out_dir_, rgb_to_id_, get_logger());

  RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 2000,
                       "Saved %s and %s | classes so far: %u",
                       image_path.c_str(), mask_path.c_str(),
                       static_cast<unsigned>(rgb_to_id_.size()));

  ++frame_counter_;
}

}  // namespace vortex::camera_segmentation
