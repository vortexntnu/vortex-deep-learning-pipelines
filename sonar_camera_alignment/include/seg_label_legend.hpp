#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>

#include <vortex_msgs/msg/seg_depth_packet.hpp>

#include <string>
#include <vector>
#include <memory>
#include <mutex>

class SegLabelLegendNode : public rclcpp::Node
{
public:
    SegLabelLegendNode();

private:
    using Packet = vortex_msgs::msg::SegDepthPacket;

    void onPacket(const Packet::ConstSharedPtr msg);
    bool extractIds(const Packet &pkt, std::vector<uint32_t> &ids);
    std::string formatLabel(uint32_t id) const;
    bool writeJson(const std::vector<uint32_t> &ids);

    std::string packet_topic_;
    bool ignore_zero_label_;
    int log_period_ms_;
    int max_ids_in_log_;
    int write_period_ms_;
    std::string json_output_path_;

    std::vector<std::string> label_names_;

    rclcpp::Subscription<Packet>::SharedPtr sub_;

    rclcpp::Time last_log_time_;
    rclcpp::Time last_write_time_;
    std::vector<uint32_t> last_ids_;
    std::vector<uint32_t> last_written_ids_;
    bool wrote_once_{false};
};
