#include "sonar_camera_alignment.hpp"

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using std::placeholders::_1;
using std::placeholders::_2;

static inline bool isFinite(float v) { return std::isfinite(v); }

SonarCameraAlignmentNode::SonarCameraAlignmentNode()
    : Node("sonar_camera_alignment_projector"),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_)
{
    packet_topic_ = declare_parameter<std::string>("packet_topic", "/synced/seg_depth_packet");
    sonar_topic_ = declare_parameter<std::string>("sonar_topic", "/front_sonar/display_mono");
    sonar_info_topic_ = declare_parameter<std::string>("sonar_info_topic", "/front_sonar/sonar_info");

    camera_frame_override_ = declare_parameter<std::string>("camera_frame_override", ""); // seg frame override
    depth_frame_override_ = declare_parameter<std::string>("depth_frame_override", "");
    sonar_frame_override_ = declare_parameter<std::string>("sonar_frame_override", "");

    sample_step_ = declare_parameter<int>("sample_step", 2); // iterate depth pixels (2..6 recommended)
    out_width_ = declare_parameter<int>("out_width", -1);
    out_height_ = declare_parameter<int>("out_height", -1);
    use_vertical_fov_filter_ = declare_parameter<bool>("use_vertical_fov_filter", true);

    projection_mode_ = declare_parameter<std::string>("projection_mode", "fan");
    invert_range_axis_ = declare_parameter<bool>("invert_range_axis", true);
    invert_angle_axis_ = declare_parameter<bool>("invert_angle_axis", false);
    horizontal_fov_rad_ = declare_parameter<double>("horizontal_fov_rad", 130.0 * M_PI / 180.0);

    refine_with_sonar_intensity_ = declare_parameter<bool>("refine_with_sonar_intensity", true);
    use_otsu_threshold_ = declare_parameter<bool>("use_otsu_threshold", true);
    intensity_threshold_ = declare_parameter<int>("intensity_threshold", 40);
    acoustic_open_iter_ = declare_parameter<int>("acoustic_open_iter", 1);
    acoustic_close_iter_ = declare_parameter<int>("acoustic_close_iter", 2);

    use_fan_valid_mask_ = declare_parameter<bool>("use_fan_valid_mask", true);
    fan_valid_threshold_ = declare_parameter<int>("fan_valid_threshold", 5);

    filter_to_single_label_ = declare_parameter<bool>("filter_to_single_label", true);
    target_label_id_ = declare_parameter<int>("target_label_id", 7);

    projected_pub_ = image_transport::create_publisher(this, "/front_sonar/projected_seg");

    sonar_info_plain_sub_ = create_subscription<SonarInfo>(
        sonar_info_topic_, rclcpp::QoS(1).reliable(),
        [this](SonarInfo::ConstSharedPtr msg)
        {
            std::lock_guard<std::mutex> lk(info_mutex_);
            last_sonar_info_ = msg;
        });

    rclcpp::QoS qos = rclcpp::QoS(rclcpp::KeepLast(10))
                          .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);

    pkt_sub_.subscribe(this, packet_topic_, qos.get_rmw_qos_profile());
    sonar_img_sub_.subscribe(this, sonar_topic_, qos.get_rmw_qos_profile());

    using Policy = message_filters::sync_policies::ApproximateTime<SegDepthPacket, Image>;
    sync_ = std::make_shared<message_filters::Synchronizer<Policy>>(Policy(80), pkt_sub_, sonar_img_sub_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.75));
    sync_->registerCallback(std::bind(&SonarCameraAlignmentNode::callback2, this, _1, _2));

    RCLCPP_INFO(get_logger(), "READY. packet=%s sonar=%s sonar_info=%s",
                packet_topic_.c_str(), sonar_topic_.c_str(), sonar_info_topic_.c_str());
}

void SonarCameraAlignmentNode::callback2(
    const SegDepthPacket::ConstSharedPtr pkt_msg,
    const Image::ConstSharedPtr sonar_img_msg)
{
    // 1) SonarInfo cache
    SonarInfo::ConstSharedPtr sonar_info;
    {
        std::lock_guard<std::mutex> lk(info_mutex_);
        sonar_info = last_sonar_info_;
    }
    if (!sonar_info)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for sonar_info...");
        return;
    }

    // 2) Frames (seg from seg image header, depth from depth_camera_info, sonar from sonar image header)
    const std::string seg_frame =
        !camera_frame_override_.empty() ? camera_frame_override_
                                        : pkt_msg->seg.header.frame_id;

    const std::string depth_frame =
        !depth_frame_override_.empty() ? depth_frame_override_
                                       : pkt_msg->depth_camera_info.header.frame_id;

    const std::string sonar_frame =
        !sonar_frame_override_.empty() ? sonar_frame_override_
                                       : sonar_img_msg->header.frame_id;

    if (seg_frame.empty() || depth_frame.empty() || sonar_frame.empty())
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Empty frame_id: seg='%s' depth='%s' sonar='%s'",
                             seg_frame.c_str(), depth_frame.c_str(), sonar_frame.c_str());
        return;
    }

    // 3) TF lookups at packet time
    geometry_msgs::msg::TransformStamped T_seg_from_depth;
    geometry_msgs::msg::TransformStamped T_sonar_from_depth;
    try
    {
        T_seg_from_depth = tf_buffer_.lookupTransform(
            seg_frame, depth_frame, pkt_msg->header.stamp, rclcpp::Duration::from_seconds(0.1));

        T_sonar_from_depth = tf_buffer_.lookupTransform(
            sonar_frame, depth_frame, pkt_msg->header.stamp, rclcpp::Duration::from_seconds(0.1));
    }
    catch (const tf2::TransformException &ex)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF lookup failed: %s", ex.what());
        return;
    }

    // 4) Seg image as label IDs
    cv::Mat seg;
    try
    {
        // Seg is usually 16UC1; we use the same encoding from the message
        seg = cv_bridge::toCvShare(pkt_msg->seg, pkt_msg, pkt_msg->seg.encoding)->image;
    }
    catch (const std::exception &e)
    {
        RCLCPP_WARN(get_logger(), "cv_bridge seg error: %s", e.what());
        return;
    }

    cv::Mat seg_mono = seg;
    if (seg.channels() != 1)
    {
        cv::cvtColor(seg, seg_mono, cv::COLOR_BGR2GRAY);
    }

    const bool seg_is_16u = (seg_mono.type() == CV_16UC1);
    if (!seg_is_16u && seg_mono.type() != CV_8UC1)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Unexpected seg type=%d (expected CV_16UC1 or CV_8UC1)", seg_mono.type());
        return;
    }

    // 5) Depth array sanity
    const uint32_t Wd = pkt_msg->width;
    const uint32_t Hd = pkt_msg->height;

    if (Wd == 0 || Hd == 0 || pkt_msg->depth_m.size() != (size_t)Wd * (size_t)Hd)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Invalid depth_m size. width=%u height=%u depth_m=%zu",
                             Wd, Hd, pkt_msg->depth_m.size());
        return;
    }

    // 6) Intrinsics (IMPORTANT: now we use BOTH intrinsics correctly)
    const auto &seg_ci = pkt_msg->seg_camera_info;
    const auto &dep_ci = pkt_msg->depth_camera_info;

    const double fxs = seg_ci.k[0], fys = seg_ci.k[4], cxs = seg_ci.k[2], cys = seg_ci.k[5];
    const double fxd = dep_ci.k[0], fyd = dep_ci.k[4], cxd = dep_ci.k[2], cyd = dep_ci.k[5];

    if (fxs <= 1e-9 || fys <= 1e-9 || fxd <= 1e-9 || fyd <= 1e-9)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Invalid intrinsics in seg/depth camera_info.");
        return;
    }

    // 7) Output size (fan reference)
    const int sonar_w = (out_width_ > 0) ? out_width_ : (int)sonar_img_msg->width;
    const int sonar_h = (out_height_ > 0) ? out_height_ : (int)sonar_img_msg->height;

    cv::Mat out = cv::Mat::zeros(sonar_h, sonar_w, seg_is_16u ? CV_16UC1 : CV_8UC1);

    const double half_vfov = 0.5 * sonar_info->vertical_fov;
    const double half_hfov = 0.5 * horizontal_fov_rad_;
    if (projection_mode_ == "fan" && half_hfov <= 1e-6)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "horizontal_fov_rad too small. Set sonar HFOV in radians.");
        return;
    }

    auto depth_at = [&](int v, int u) -> float
    {
        return pkt_msg->depth_m[(size_t)v * (size_t)Wd + (size_t)u];
    };

    auto isFinite = [](float v)
    { return std::isfinite(v); };

    int projected_points = 0;

    // 8) Iterate depth pixels (DEPTH GRID), transform to seg (for label sampling), then to sonar
    const int step = std::max(1, sample_step_);
    for (int vd = 0; vd < (int)Hd; vd += step)
    {
        for (int ud = 0; ud < (int)Wd; ud += step)
        {
            const float Z = depth_at(vd, ud);
            if (!isFinite(Z) || Z <= 0.0f)
                continue;

            // Back-project in DEPTH camera frame (uses depth intrinsics)
            const double Xd = ((double)ud - cxd) * (double)Z / fxd;
            const double Yd = ((double)vd - cyd) * (double)Z / fyd;
            const double Zd = (double)Z;

            geometry_msgs::msg::PointStamped p_depth;
            p_depth.header.stamp = pkt_msg->header.stamp;
            p_depth.header.frame_id = depth_frame;
            p_depth.point.x = Xd;
            p_depth.point.y = Yd;
            p_depth.point.z = Zd;

            // Transform depth point -> seg frame, then project to seg image (uses seg intrinsics)
            geometry_msgs::msg::PointStamped p_seg;
            tf2::doTransform(p_depth, p_seg, T_seg_from_depth);

            const double Xs = p_seg.point.x;
            const double Ys = p_seg.point.y;
            const double Zs = p_seg.point.z;
            if (Zs <= 1e-6)
                continue;

            const int us = (int)std::lround(fxs * (Xs / Zs) + cxs);
            const int vs = (int)std::lround(fys * (Ys / Zs) + cys);

            if (vs < 0 || vs >= seg_mono.rows || us < 0 || us >= seg_mono.cols)
                continue;

            const uint32_t label = seg_is_16u
                                       ? (uint32_t)seg_mono.at<uint16_t>(vs, us)
                                       : (uint32_t)seg_mono.at<uint8_t>(vs, us);

            if (label == 0)
                continue;
            if (filter_to_single_label_ && (int)label != target_label_id_)
                continue;

            // Transform depth point -> sonar frame (same original point, different TF)
            geometry_msgs::msg::PointStamped p_sonar;
            tf2::doTransform(p_depth, p_sonar, T_sonar_from_depth);

            const double x = p_sonar.point.x;
            const double y = p_sonar.point.y;
            const double z = p_sonar.point.z;

            const double r_xy = std::sqrt(x * x + y * y);
            if (r_xy < sonar_info->min_range || r_xy > sonar_info->max_range)
                continue;

            if (use_vertical_fov_filter_)
            {
                const double elev = std::atan2(z, std::max(1e-9, r_xy));
                if (std::fabs(elev) > half_vfov)
                    continue;
            }

            int ui = -1, vi = -1;

            if (projection_mode_ == "fan")
            {
                double ang = std::atan2(y, x);
                if (invert_angle_axis_)
                    ang = -ang;
                if (ang < -half_hfov || ang > half_hfov)
                    continue;

                const double u_norm = (ang + half_hfov) / (2.0 * half_hfov);
                const double denom = std::max(1e-9, (sonar_info->max_range - sonar_info->min_range));
                const double v_norm = (r_xy - sonar_info->min_range) / denom;

                ui = (int)std::lround(u_norm * (sonar_w - 1));
                vi = (int)std::lround(v_norm * (sonar_h - 1));

                if (invert_range_axis_)
                    vi = (sonar_h - 1) - vi;
            }
            else
            {
                continue; // cartesian mode not implemented here
            }

            if (ui < 0 || ui >= sonar_w || vi < 0 || vi >= sonar_h)
                continue;

            if (seg_is_16u)
                out.at<uint16_t>(vi, ui) = (uint16_t)label;
            else
                out.at<uint8_t>(vi, ui) = (uint8_t)label;

            projected_points++;
        }
    }

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Publishing projected_seg (points=%d)", projected_points);

    cv::dilate(out, out, cv::Mat(), cv::Point(-1, -1), 1);

    // 9) Optional masks using sonar intensity
    cv::Mat sonar_img;
    try
    {
        sonar_img = cv_bridge::toCvShare(sonar_img_msg)->image;
    }
    catch (const std::exception &e)
    {
        RCLCPP_WARN(get_logger(), "cv_bridge sonar error: %s", e.what());
        return;
    }

    if (use_fan_valid_mask_)
    {
        const int thr_val = std::max(0, fan_valid_threshold_);
        cv::Mat fan_valid = (sonar_img > thr_val);
        out.setTo(0, ~fan_valid);
    }

    if (refine_with_sonar_intensity_)
    {
        cv::Mat sonar_mono8;
        if (sonar_img_msg->encoding == "mono8")
            sonar_mono8 = sonar_img;
        else if (sonar_img.type() == CV_8UC1)
            sonar_mono8 = sonar_img;
        else
            sonar_img.convertTo(sonar_mono8, CV_8U);

        cv::Mat acoustic_mask;
        if (use_otsu_threshold_)
            cv::threshold(sonar_mono8, acoustic_mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        else
        {
            const int t = std::max(0, std::min(255, intensity_threshold_));
            cv::threshold(sonar_mono8, acoustic_mask, t, 255, cv::THRESH_BINARY);
        }

        if (acoustic_open_iter_ > 0)
            cv::morphologyEx(acoustic_mask, acoustic_mask, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), acoustic_open_iter_);
        if (acoustic_close_iter_ > 0)
            cv::morphologyEx(acoustic_mask, acoustic_mask, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), acoustic_close_iter_);

        cv::Mat keep = (acoustic_mask > 0);
        out.setTo(0, ~keep);
    }

    // 10) Publish projected segmentation in sonar image space
    sensor_msgs::msg::Image out_msg;
    const std::string out_encoding = seg_is_16u ? "16UC1" : "mono8";
    cv_bridge::CvImage cv_out(pkt_msg->header, out_encoding, out);
    cv_out.toImageMsg(out_msg);

    // IMPORTANT: match sonar timestamp & frame
    out_msg.header.stamp = sonar_img_msg->header.stamp;
    out_msg.header.frame_id = sonar_frame;

    projected_pub_.publish(out_msg);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SonarCameraAlignmentNode>());
    rclcpp::shutdown();
    return 0;
}
