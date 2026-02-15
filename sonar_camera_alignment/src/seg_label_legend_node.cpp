#include "seg_label_legend.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>

SegLabelLegendNode::SegLabelLegendNode()
    : Node("seg_label_legend"),
      last_log_time_(this->get_clock()->now()),
      last_write_time_(this->get_clock()->now())
{
    packet_topic_ = declare_parameter<std::string>("packet_topic", "/synced/seg_depth_packet");
    ignore_zero_label_ = declare_parameter<bool>("ignore_zero_label", true);
    log_period_ms_ = declare_parameter<int>("log_period_ms", 1000);
    max_ids_in_log_ = declare_parameter<int>("max_ids_in_log", 32);
    write_period_ms_ = declare_parameter<int>("write_period_ms", 1000);
    json_output_path_ = declare_parameter<std::string>("json_output_path", "");

    label_names_ = declare_parameter<std::vector<std::string>>("label_names",
                                                               std::vector<std::string>{});

    sub_ = create_subscription<Packet>(
        packet_topic_, rclcpp::QoS(10),
        std::bind(&SegLabelLegendNode::onPacket, this, std::placeholders::_1));

    if (json_output_path_.empty())
    {
        const char *home = std::getenv("HOME");
        if (home && std::string(home).size() > 0)
        {
            json_output_path_ = std::string(home) + "/seg_label_legend.json";
        }
        else
        {
            json_output_path_ = "/tmp/seg_label_legend.json";
        }
    }

    RCLCPP_INFO(get_logger(), "SegLabelLegend ready. packet=%s json=%s",
                packet_topic_.c_str(), json_output_path_.c_str());
}

void SegLabelLegendNode::onPacket(const Packet::ConstSharedPtr msg)
{
    std::vector<uint32_t> ids;
    if (!extractIds(*msg, ids))
    {
        return;
    }

    if (ids != last_ids_)
    {
        const auto now = this->get_clock()->now();
        if (log_period_ms_ <= 0 ||
            (now - last_log_time_) >= rclcpp::Duration::from_nanoseconds(
                static_cast<int64_t>(log_period_ms_) * 1000000LL))
        {
            std::ostringstream oss;
            oss << "Seg labels (" << ids.size() << "): ";
            const size_t max_log = (max_ids_in_log_ <= 0) ? ids.size()
                                                          : std::min(ids.size(), static_cast<size_t>(max_ids_in_log_));
            for (size_t i = 0; i < max_log; ++i)
            {
                if (i > 0)
                    oss << ", ";
                oss << formatLabel(ids[i]);
            }
            if (max_log < ids.size())
                oss << " ...";
            RCLCPP_INFO(get_logger(), "%s", oss.str().c_str());
            last_log_time_ = now;
        }
        last_ids_ = ids;
    }

    writeJson(ids);
}

bool SegLabelLegendNode::extractIds(const Packet &pkt, std::vector<uint32_t> &ids)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(pkt.seg, pkt.seg.encoding);
    }
    catch (const std::exception &e)
    {
        RCLCPP_WARN(get_logger(), "cv_bridge seg error: %s", e.what());
        return false;
    }

    const std::string &enc = pkt.seg.encoding;
    const cv::Mat &img = cv_ptr->image;

    if (enc == "mono8" || enc == "8UC1")
    {
        std::array<bool, 256> seen{};
        for (int r = 0; r < img.rows; ++r)
        {
            const uint8_t *row = img.ptr<uint8_t>(r);
            for (int c = 0; c < img.cols; ++c)
            {
                const uint8_t v = row[c];
                if (ignore_zero_label_ && v == 0)
                    continue;
                if (!seen[v])
                {
                    seen[v] = true;
                    ids.push_back(static_cast<uint32_t>(v));
                }
            }
        }
    }
    else if (enc == "mono16" || enc == "16UC1")
    {
        std::vector<bool> seen(65536, false);
        for (int r = 0; r < img.rows; ++r)
        {
            const uint16_t *row = img.ptr<uint16_t>(r);
            for (int c = 0; c < img.cols; ++c)
            {
                const uint16_t v = row[c];
                if (ignore_zero_label_ && v == 0)
                    continue;
                if (!seen[v])
                {
                    seen[v] = true;
                    ids.push_back(static_cast<uint32_t>(v));
                }
            }
        }
    }
    else
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "Unsupported seg encoding: %s (expected mono8/mono16)", enc.c_str());
        return false;
    }

    std::sort(ids.begin(), ids.end());
    return true;
}

bool SegLabelLegendNode::writeJson(const std::vector<uint32_t> &ids)
{
    const auto now = this->get_clock()->now();
    // Always write at least once as soon as we receive a packet.
    if (wrote_once_ && write_period_ms_ > 0 &&
        (now - last_write_time_) < rclcpp::Duration::from_nanoseconds(
            static_cast<int64_t>(write_period_ms_) * 1000000LL))
    {
        return false;
    }

    if (wrote_once_ && ids == last_written_ids_)
    {
        return false;
    }

    std::ofstream out(json_output_path_, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "Failed to write JSON: %s (errno=%d: %s)",
                             json_output_path_.c_str(), errno, std::strerror(errno));
        return false;
    }

    out << "{\n  \"ids\": [";
    for (size_t i = 0; i < ids.size(); ++i)
    {
        if (i > 0)
            out << ", ";
        out << ids[i];
    }
    out << "],\n  \"labels\": [";
    bool first = true;
    for (size_t i = 0; i < ids.size(); ++i)
    {
        const uint32_t id = ids[i];
        if (id < label_names_.size() && !label_names_[id].empty())
        {
            if (!first)
                out << ", ";
            out << "{\"id\": " << id << ", \"name\": \""
                << label_names_[id] << "\"}";
            first = false;
        }
    }
    out << "]\n}\n";

    last_write_time_ = now;
    last_written_ids_ = ids;
    wrote_once_ = true;
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Wrote seg label JSON: %s", json_output_path_.c_str());
    return true;
}

std::string SegLabelLegendNode::formatLabel(uint32_t id) const
{
    std::ostringstream oss;
    oss << id;
    if (id < label_names_.size() && !label_names_[id].empty())
    {
        oss << " (" << label_names_[id] << ")";
    }
    return oss.str();
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SegLabelLegendNode>());
    rclcpp::shutdown();
    return 0;
}
