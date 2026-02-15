#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <vortex_msgs/msg/seg_depth_packet.hpp>

#include <atomic>
#include <mutex>

class SegDepthSyncNode : public rclcpp::Node
{
public:
    using Image = sensor_msgs::msg::Image;
    using CameraInfo = sensor_msgs::msg::CameraInfo;

    SegDepthSyncNode();

private:
    void cb(const Image::ConstSharedPtr seg_msg,
            const Image::ConstSharedPtr depth_msg);

    // Topics
    std::string seg_topic_;
    std::string depth_topic_;
    std::string seg_cam_info_topic_;
    std::string depth_cam_info_topic_;
    std::string out_topic_;

    // Sync params
    int sync_queue_size_{400};
    double max_sync_interval_s_{5.0};

    // Depth conversion params
    double depth_scale_{0.001};
    double max_depth_m_{50.0};
    bool keep_zero_depth_{false};

    // CameraInfo cache
    std::mutex cam_info_mtx_;
    CameraInfo::ConstSharedPtr last_seg_cam_info_;
    CameraInfo::ConstSharedPtr last_depth_cam_info_;

    rclcpp::Subscription<CameraInfo>::SharedPtr seg_cam_info_sub_;
    rclcpp::Subscription<CameraInfo>::SharedPtr depth_cam_info_sub_;

    // Publisher
    rclcpp::Publisher<vortex_msgs::msg::SegDepthPacket>::SharedPtr pub_;

    // message_filters
    message_filters::Subscriber<Image> seg_sub_;
    message_filters::Subscriber<Image> depth_sub_;

    using Policy = message_filters::sync_policies::ApproximateTime<Image, Image>;
    std::shared_ptr<message_filters::Synchronizer<Policy>> sync_;

    // Stats
    std::atomic<uint64_t> seg_rx_{0};
    std::atomic<uint64_t> depth_rx_{0};
    std::atomic<uint64_t> pkt_tx_{0};

    rclcpp::TimerBase::SharedPtr status_timer_;
};
