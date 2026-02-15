#include "sonar_projected_overlay.hpp"

#include <cmath>
#include <cstdint>

using std::placeholders::_1;
using std::placeholders::_2;

static inline cv::Vec3b hsvToBgr(float h_deg, float s, float v)
{
    // h_deg: [0,360)
    const float c = v * s;
    const float hp = std::fmod(h_deg / 60.0f, 6.0f);
    const float x = c * (1.0f - std::fabs(std::fmod(hp, 2.0f) - 1.0f));
    float r = 0, g = 0, b = 0;
    if (0.0f <= hp && hp < 1.0f)
    {
        r = c;
        g = x;
    }
    else if (1.0f <= hp && hp < 2.0f)
    {
        r = x;
        g = c;
    }
    else if (2.0f <= hp && hp < 3.0f)
    {
        g = c;
        b = x;
    }
    else if (3.0f <= hp && hp < 4.0f)
    {
        g = x;
        b = c;
    }
    else if (4.0f <= hp && hp < 5.0f)
    {
        r = x;
        b = c;
    }
    else
    {
        r = c;
        b = x;
    }
    const float m = v - c;
    const uint8_t R = static_cast<uint8_t>(std::lround((r + m) * 255.0f));
    const uint8_t G = static_cast<uint8_t>(std::lround((g + m) * 255.0f));
    const uint8_t B = static_cast<uint8_t>(std::lround((b + m) * 255.0f));
    return cv::Vec3b(B, G, R);
}

static inline cv::Vec3b colorForLabel(uint32_t label)
{
    // Deterministic, “spread out” hue via golden angle.
    const float hue = std::fmod(static_cast<float>(label) * 137.50776405f, 360.0f);
    return hsvToBgr(hue, 0.95f, 1.0f);
}

SonarProjectedOverlayNode::SonarProjectedOverlayNode()
    : Node("sonar_projected_overlay")
{
    sonar_topic_ = declare_parameter<std::string>("sonar_topic", "/front_sonar/display_mono");
    projected_topic_ = declare_parameter<std::string>("projected_topic", "/front_sonar/projected_seg");
    overlay_topic_ = declare_parameter<std::string>("overlay_topic", "/front_sonar/overlay");

    alpha_ = declare_parameter<double>("alpha", 0.65);
    dilate_iter_ = declare_parameter<int>("dilate_iter", 1);
    colorize_labels_ = declare_parameter<bool>("colorize_labels", false);
    mask_to_sonar_fan_ = declare_parameter<bool>("mask_to_sonar_fan", true);
    fan_mask_threshold_ = declare_parameter<int>("fan_mask_threshold", 5);

    // If true, resize projected_seg to match sonar size (debug-only).
    resize_projected_to_sonar_ = declare_parameter<bool>("resize_projected_to_sonar", false);

    overlay_pub_ = image_transport::create_publisher(this, overlay_topic_);

    // Use SensorDataQoS to be compatible with simulator streams.
    auto qos = rclcpp::SensorDataQoS();

    sonar_sub_.subscribe(this, sonar_topic_, qos.get_rmw_qos_profile());
    proj_sub_.subscribe(this, projected_topic_, qos.get_rmw_qos_profile());

    using Policy = message_filters::sync_policies::ApproximateTime<Image, Image>;
    sync_ = std::make_shared<message_filters::Synchronizer<Policy>>(
        Policy(50), sonar_sub_, proj_sub_);

    // projected_seg can be slower; allow larger sync window
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(2.5));

    sync_->registerCallback(std::bind(&SonarProjectedOverlayNode::cb, this, _1, _2));

    RCLCPP_INFO(get_logger(), "Overlay node ready.");
    RCLCPP_INFO(get_logger(), " sonar_topic     : %s", sonar_topic_.c_str());
    RCLCPP_INFO(get_logger(), " projected_topic : %s", projected_topic_.c_str());
    RCLCPP_INFO(get_logger(), " overlay_topic   : %s", overlay_topic_.c_str());
    RCLCPP_INFO(get_logger(), " colorize_labels : %s", colorize_labels_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), " mask_to_sonar_fan: %s (thr=%d)",
                mask_to_sonar_fan_ ? "true" : "false", fan_mask_threshold_);
}

void SonarProjectedOverlayNode::cb(const Image::ConstSharedPtr sonar_msg,
                                   const Image::ConstSharedPtr proj_msg)
{
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "cb: sonar=%dx%d (%s) proj=%dx%d (%s)",
                         sonar_msg->width, sonar_msg->height, sonar_msg->encoding.c_str(),
                         proj_msg->width, proj_msg->height, proj_msg->encoding.c_str());

    cv::Mat sonar, proj;
    try
    {
        sonar = cv_bridge::toCvShare(sonar_msg)->image;
        proj = cv_bridge::toCvShare(proj_msg)->image;
    }
    catch (const std::exception &e)
    {
        RCLCPP_WARN(get_logger(), "cv_bridge error: %s", e.what());
        return;
    }

    // Ensure same size (required for pixelwise overlay).
    if (sonar.cols != proj.cols || sonar.rows != proj.rows)
    {
        if (!resize_projected_to_sonar_)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "Size mismatch: sonar=%dx%d proj=%dx%d. "
                                 "Fix topics so they match, or set resize_projected_to_sonar:=true (debug).",
                                 sonar.cols, sonar.rows, proj.cols, proj.rows);
            return;
        }

        // Debug fallback: resize projected_seg to sonar size using nearest neighbor (keeps labels).
        cv::Mat proj_resized;
        cv::resize(proj, proj_resized, cv::Size(sonar.cols, sonar.rows), 0, 0, cv::INTER_NEAREST);
        proj = proj_resized;
    }

    // Convert sonar to mono8 for display base.
    cv::Mat sonar_mono8;
    if (sonar_msg->encoding == "mono8")
    {
        sonar_mono8 = sonar;
    }
    else if (sonar_msg->encoding == "16UC1")
    {
        double minv, maxv;
        cv::minMaxLoc(sonar, &minv, &maxv);
        if (maxv < 1e-9)
            maxv = 1.0;
        sonar.convertTo(sonar_mono8, CV_8U, 255.0 / maxv);
    }
    else if (sonar.channels() == 3)
    {
        cv::cvtColor(sonar, sonar_mono8, cv::COLOR_BGR2GRAY);
    }
    else
    {
        sonar.convertTo(sonar_mono8, CV_8U);
    }

    // Base in BGR.
    cv::Mat base_bgr;
    cv::cvtColor(sonar_mono8, base_bgr, cv::COLOR_GRAY2BGR);

    // Build mask from projected_seg: any label > 0.
    cv::Mat mask = (proj > 0);
    if (dilate_iter_ > 0)
    {
        cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), dilate_iter_);
    }

    // Create overlay image.
    cv::Mat overlay = base_bgr.clone();
    if (!colorize_labels_)
    {
        // Red overlay for any mask pixel.
        overlay.setTo(cv::Scalar(0, 0, 255), mask);
    }
    else
    {
        // Color each label ID deterministically (supports mono8 and mono16 projected_seg).
        if (proj.type() == CV_8UC1)
        {
            for (int y = 0; y < proj.rows; ++y)
            {
                const uint8_t *p = proj.ptr<uint8_t>(y);
                cv::Vec3b *o = overlay.ptr<cv::Vec3b>(y);
                for (int x = 0; x < proj.cols; ++x)
                {
                    const uint32_t label = static_cast<uint32_t>(p[x]);
                    if (label == 0 || label == 65534)
                        continue;
                    o[x] = colorForLabel(label);
                }
            }
        }
        else if (proj.type() == CV_16UC1)
        {
            for (int y = 0; y < proj.rows; ++y)
            {
                const uint16_t *p = proj.ptr<uint16_t>(y);
                cv::Vec3b *o = overlay.ptr<cv::Vec3b>(y);
                for (int x = 0; x < proj.cols; ++x)
                {
                    const uint32_t label = static_cast<uint32_t>(p[x]);
                    if (label == 0)
                        continue;
                    o[x] = colorForLabel(label);
                }
            }
        }
        else
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                                 "projected_seg has unsupported type=%d (expected mono8/mono16). "
                                 "Falling back to red mask overlay.",
                                 proj.type());
            overlay.setTo(cv::Scalar(0, 0, 255), mask);
        }
    }

    // Alpha blend so you still see sonar below.
    cv::Mat out;
    cv::addWeighted(overlay, alpha_, base_bgr, 1.0 - alpha_, 0.0, out);

    // Optional: restrict overlay to the visible fan region (avoid painting black outside).
    if (mask_to_sonar_fan_)
    {
        const int thr = std::max(0, std::min(255, fan_mask_threshold_));
        cv::Mat fan_valid = (sonar_mono8 > thr);
        // Outside the fan, show only the sonar base (no colored overlay).
        base_bgr.copyTo(out, ~fan_valid);
    }

    // Publish
    sensor_msgs::msg::Image out_msg;
    cv_bridge::CvImage cv_out(sonar_msg->header, "bgr8", out);
    cv_out.toImageMsg(out_msg);

    out_msg.header.stamp = sonar_msg->header.stamp;
    out_msg.header.frame_id = sonar_msg->header.frame_id;

    overlay_pub_.publish(out_msg);
}

// -------- main() is REQUIRED because this is an executable --------
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SonarProjectedOverlayNode>());
    rclcpp::shutdown();
    return 0;
}
