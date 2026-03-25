#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/header.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <stonefish_ros2/srv/respawn.hpp>
#include <vortex/utils/math.hpp>

namespace vortex::sonar_segmentation {

class SonarSegmentationNode : public rclcpp::Node
{
public:
    SonarSegmentationNode();

private:

    void segmentationCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void sonarCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void callRespawnService();

    void declare_parameters();
    void get_parameters();
    void setup_publishers_and_subscribers();
    void setup_service_clients();
    void setup_timer();

    void process();

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr segmentation_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sonar_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

    // Sercice Clients
    rclcpp::Client<stonefish_ros2::srv::Respawn>::SharedPtr respawn_client_;

    // Timers
    rclcpp::TimerBase::SharedPtr respawn_timer_;

    // Publisher
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr output_overlay_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr output_pub_;

    // Latest frames
    cv::Mat segmentation_img_;
    cv::Mat depth_img_;
    cv::Mat sonar_img_;

    bool segmentation_ready_ = false;
    bool depth_ready_ = false;
    bool sonar_ready_ = false;
    bool camera_info_ready_ = false;
    bool timer_ready_ = false;

    // Parameters
    double fov_ = 0.0;
    double sonar_range_ = 0.0;
    cv::Mat camera_k_;
    std::string output_frame_id_;
};

} // namespace vortex::sonar_segmentation
