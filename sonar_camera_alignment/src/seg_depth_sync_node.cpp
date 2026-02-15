#include "seg_depth_sync.hpp"

#include <rmw/qos_profiles.h>
#include <limits>
#include <cmath>
#include <mutex>

#include <cv_bridge/cv_bridge.h>

using std::placeholders::_1;
using std::placeholders::_2;

SegDepthSyncNode::SegDepthSyncNode()
    : Node("seg_depth_sync")
{
    // Inputs
    seg_topic_ = declare_parameter<std::string>("seg_topic", "/front_camera_seg/image_raw");
    depth_topic_ = declare_parameter<std::string>("depth_topic", "/depth_camera/image_depth");

    seg_cam_info_topic_ = declare_parameter<std::string>("seg_camera_info_topic", "/front_camera_seg/camera_info");
    depth_cam_info_topic_ = declare_parameter<std::string>("depth_camera_info_topic", "/depth_camera/camera_info");

    // Output
    out_topic_ = declare_parameter<std::string>("out_topic", "/synced/seg_depth_packet2");

    // Sync tuning
    sync_queue_size_ = declare_parameter<int>("sync_queue_size", 400);
    max_sync_interval_s_ = declare_parameter<double>("max_sync_interval_s", 5.0);

    // Depth conversion
    depth_scale_ = declare_parameter<double>("depth_scale", 0.001); // for 16UC1, mm->m
    max_depth_m_ = declare_parameter<double>("max_depth_m", 50.0);
    keep_zero_depth_ = declare_parameter<bool>("keep_zero_depth", false);

    // Cache seg camera_info
    seg_cam_info_sub_ = create_subscription<CameraInfo>(
        seg_cam_info_topic_, rclcpp::QoS(1).reliable(),
        [this](CameraInfo::ConstSharedPtr msg)
        {
            std::lock_guard<std::mutex> lk(cam_info_mtx_);
            last_seg_cam_info_ = msg;
        });
    // Cache depth camera_info
    depth_cam_info_sub_ = create_subscription<CameraInfo>(
        depth_cam_info_topic_, rclcpp::QoS(1).reliable(),
        [this](CameraInfo::ConstSharedPtr msg)
        {
            std::lock_guard<std::mutex> lk(cam_info_mtx_);
            last_depth_cam_info_ = msg;
        });

    // Publisher
    pub_ = create_publisher<vortex_msgs::msg::SegDepthPacket>(out_topic_, rclcpp::QoS(10).reliable());

    // Sync seg + depth
    rclcpp::QoS qos = rclcpp::QoS(rclcpp::KeepLast(20))
                          .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);

    seg_sub_.subscribe(this, seg_topic_, qos.get_rmw_qos_profile());
    depth_sub_.subscribe(this, depth_topic_, qos.get_rmw_qos_profile());

    using Policy = message_filters::sync_policies::ApproximateTime<Image, Image>;
    sync_ = std::make_shared<message_filters::Synchronizer<Policy>>(
        Policy(sync_queue_size_), seg_sub_, depth_sub_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(max_sync_interval_s_));
    sync_->registerCallback(std::bind(&SegDepthSyncNode::cb, this, _1, _2));

    status_timer_ = create_wall_timer(
        std::chrono::seconds(2),
        [this]()
        {
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "Status: seg_rx=%lu depth_rx=%lu pkt_tx=%lu (queue=%d max_dt=%.2fs)",
                                 (unsigned long)seg_rx_.load(),
                                 (unsigned long)depth_rx_.load(),
                                 (unsigned long)pkt_tx_.load(),
                                 sync_queue_size_,
                                 max_sync_interval_s_);
        });

    RCLCPP_INFO(get_logger(), "SegDepthPacket publisher ready.");
    RCLCPP_INFO(get_logger(), "  seg:              %s", seg_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  depth:            %s", depth_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  seg_camera_info:  %s", seg_cam_info_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  depth_camera_info:%s", depth_cam_info_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  out:              %s (vortex_msgs/msg/SegDepthPacket)", out_topic_.c_str());
}

void SegDepthSyncNode::cb(const Image::ConstSharedPtr seg_msg,
                          const Image::ConstSharedPtr depth_msg)
{
    seg_rx_.fetch_add(1);
    depth_rx_.fetch_add(1);

    // Get latest intrinsics
    CameraInfo::ConstSharedPtr seg_ci, depth_ci;
    {
        std::lock_guard<std::mutex> lk(cam_info_mtx_);
        seg_ci = last_seg_cam_info_;
        depth_ci = last_depth_cam_info_;
    }
    if (!seg_ci || !depth_ci)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for seg/depth camera_info...");
        return;
    }

    // Convert depth image to cv::Mat
    cv::Mat depth;
    try
    {
        // keep native encoding here; we convert below
        depth = cv_bridge::toCvShare(depth_msg)->image;
    }
    catch (const std::exception &e)
    {
        RCLCPP_WARN(get_logger(), "cv_bridge depth error: %s", e.what());
        return;
    }

    const bool depth_is_32f = (depth_msg->encoding == "32FC1");
    const bool depth_is_16u = (depth_msg->encoding == "16UC1");
    if (!depth_is_32f && !depth_is_16u)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Unsupported depth encoding: %s", depth_msg->encoding.c_str());
        return;
    }

    const uint32_t W = depth_msg->width;
    const uint32_t H = depth_msg->height;

    vortex_msgs::msg::SegDepthPacket pkt;

    // Use depth stamp as master time (typical)
    pkt.header = seg_msg->header;
    pkt.header.stamp = depth_msg->header.stamp;

    pkt.seg = *seg_msg;
    pkt.seg.header.stamp = pkt.header.stamp;

    pkt.seg_camera_info = *seg_ci;
    pkt.seg_camera_info.header.stamp = pkt.header.stamp;

    pkt.depth_camera_info = *depth_ci;
    pkt.depth_camera_info.header.stamp = pkt.header.stamp;

    pkt.width = W;
    pkt.height = H;
    pkt.depth_m.resize((size_t)W * (size_t)H);

    const float nanv = std::numeric_limits<float>::quiet_NaN();

    if (depth_is_32f)
    {
        for (uint32_t v = 0; v < H; ++v)
        {
            const float *row = depth.ptr<float>((int)v);
            for (uint32_t u = 0; u < W; ++u)
            {
                float z = row[u];
                if (!std::isfinite(z) || z <= 0.0f || z > (float)max_depth_m_)
                    pkt.depth_m[(size_t)v * W + u] = keep_zero_depth_ ? 0.0f : nanv;
                else
                    pkt.depth_m[(size_t)v * W + u] = z;
            }
        }
    }
    else
    {
        const float scale = (float)depth_scale_;
        for (uint32_t v = 0; v < H; ++v)
        {
            const uint16_t *row = depth.ptr<uint16_t>((int)v);
            for (uint32_t u = 0; u < W; ++u)
            {
                uint16_t raw = row[u];
                if (raw == 0)
                {
                    pkt.depth_m[(size_t)v * W + u] = keep_zero_depth_ ? 0.0f : nanv;
                    continue;
                }
                float z = (float)raw * scale;
                if (!std::isfinite(z) || z <= 0.0f || z > (float)max_depth_m_)
                    pkt.depth_m[(size_t)v * W + u] = keep_zero_depth_ ? 0.0f : nanv;
                else
                    pkt.depth_m[(size_t)v * W + u] = z;
            }
        }
    }

    // Helpful warning: frames will often differ (that’s OK now)
    if (pkt.seg.header.frame_id != pkt.depth_camera_info.header.frame_id)
    {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Frames: seg=%s depth=%s (OK: projector handles TF)",
                             pkt.seg.header.frame_id.c_str(),
                             pkt.depth_camera_info.header.frame_id.c_str());
    }

    pub_->publish(pkt);
    pkt_tx_.fetch_add(1);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SegDepthSyncNode>());
    rclcpp::shutdown();
    return 0;
}
