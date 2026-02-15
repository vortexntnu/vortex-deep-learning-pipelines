#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <memory>

class SonarProjectedOverlayNode : public rclcpp::Node
{
public:
    SonarProjectedOverlayNode();

private:
    using Image = sensor_msgs::msg::Image;

    void cb(const Image::ConstSharedPtr sonar_msg,
            const Image::ConstSharedPtr proj_msg);

    std::string sonar_topic_;
    std::string projected_topic_;
    std::string overlay_topic_;

    double alpha_;
    int dilate_iter_;
    bool colorize_labels_;
    bool resize_projected_to_sonar_;
    bool mask_to_sonar_fan_;
    int fan_mask_threshold_;

    image_transport::Publisher overlay_pub_;

    message_filters::Subscriber<Image> sonar_sub_;
    message_filters::Subscriber<Image> proj_sub_;
    std::shared_ptr<message_filters::Synchronizer<
        message_filters::sync_policies::ApproximateTime<Image, Image>>>
        sync_;
};
