#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <stonefish_ros2/msg/sonar_info.hpp>
#include "vortex_msgs/msg/seg_depth_packet.hpp"

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <rmw/qos_profiles.h>

#include <cmath>
#include <string>
#include <memory>
#include <mutex>

class SonarCameraAlignmentNode : public rclcpp::Node
{
public:
    SonarCameraAlignmentNode();

private:
    using Image = sensor_msgs::msg::Image;
    using SonarInfo = stonefish_ros2::msg::SonarInfo;
    using SegDepthPacket = vortex_msgs::msg::SegDepthPacket;

    static inline bool isFinite(float v) { return std::isfinite(v); }

    void callback2(const SegDepthPacket::ConstSharedPtr pkt_msg,
                   const Image::ConstSharedPtr sonar_img_msg);

    // Topics
    std::string packet_topic_;
    std::string sonar_topic_;
    std::string sonar_info_topic_;
    std::string camera_frame_override_;
    std::string sonar_frame_override_;
    std::string depth_frame_override_;

    // Performance / filters (runtime-tunable)
    int sample_step_;
    bool use_vertical_fov_filter_;

    // Forced output size (-1 = use incoming sonar image size) (runtime-tunable)
    int out_width_;
    int out_height_;

    // Projection mode (runtime-tunable)
    std::string projection_mode_; // "fan" or "cartesian"
    bool invert_range_axis_;
    bool invert_angle_axis_;
    double horizontal_fov_rad_; // fan HFOV in radians

    // Cartesian mapping params (legacy mode)
    double origin_u_, origin_v_;
    double axis_u_x_, axis_u_y_, axis_v_x_, axis_v_y_;

    // Sonar-intensity refinement (pipe matching) (runtime-tunable)
    bool refine_with_sonar_intensity_;
    bool use_otsu_threshold_;
    int intensity_threshold_;
    int acoustic_open_iter_;
    int acoustic_close_iter_;

    // Fan validity mask (runtime-tunable)
    bool use_fan_valid_mask_;
    int fan_valid_threshold_;

    // Label filtering (runtime-tunable)
    bool filter_to_single_label_;
    int target_label_id_;

    // TF
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // Publisher
    image_transport::Publisher projected_pub_;

    // Sync subscribers: (SegDepthPacket, sonar image)
    message_filters::Subscriber<SegDepthPacket> pkt_sub_;
    message_filters::Subscriber<Image> sonar_img_sub_;
    using Policy = message_filters::sync_policies::ApproximateTime<SegDepthPacket, sensor_msgs::msg::Image>;
    std::shared_ptr<message_filters::Synchronizer<Policy>> sync_;

    // Cached sonar_info
    rclcpp::Subscription<SonarInfo>::SharedPtr sonar_info_plain_sub_;
    SonarInfo::ConstSharedPtr last_sonar_info_;
    std::mutex info_mutex_;

    // Runtime parameter handling
    std::mutex param_mutex_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
};
