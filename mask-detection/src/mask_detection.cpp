#include "mask_detection/mask_detection.hpp"

// For binding synchronized callback
using std::placeholders::_1;    // segmentation image color
using std::placeholders::_2;    // segmentation image id

MaskDetectionNode::MaskDetectionNode(const rclcpp::NodeOptions& options)
    : Node("mask_detection_node", options) {
    // Topics to subscribe to
    auto segmentation_image_color_sub_topic =
        this->declare_parameter<std::string>("segmentation_image_color_sub_topic");
    auto segmentation_image_id_sub_topic =
        this->declare_parameter<std::string>("segmentation_image_id_sub_topic");

    // Quality of Service settings
    rclcpp::QoS qos = rclcpp::QoS(rclcpp::KeepLast(10))
                          .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
    segmentation_image_color_sub_.subscribe(this, segmentation_image_color_sub_topic,
                               qos.get_rmw_qos_profile());
    segmentation_image_id_sub_.subscribe(this, segmentation_image_id_sub_topic,
                               qos.get_rmw_qos_profile());
    
    // Synchronization policy
    sync_ = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(
        MySyncPolicy(10), segmentation_image_color_sub_, segmentation_image_id_sub_);
    sync_->registerCallback(std::bind(
        &MaskDetectionNode::synchronized_callback, this, _1, _2));    
    // Publishers
    seg_image_color_pub_ =
        this->create_publisher<sensor_msgs::msg::Image>("segmentation_image_color", 10);
    seg_image_id_pub_ =
        this->create_publisher<sensor_msgs::msg::Image>("segmentation_image_id", 10);
    // ============ Output directory ============
    // 1) Read parameter 'output_dir' (default: $HOME/seg_frames)
    std::string default_out = [](){
        const char* home = std::getenv("HOME");
        if (home && *home) return std::string(home) + "/seg_frames";
        return std::string("/tmp/seg_frames"); // fallback if no HOME
    }();
    auto out_dir_param = this->declare_parameter<std::string>("output_dir", default_out);
    out_dir_ = std::filesystem::path(out_dir_param);

    // 2) Create directory if not existing
    std::error_code ec;
    std::filesystem::create_directories(out_dir_, ec);
    if (ec) {
        RCLCPP_WARN(get_logger(), "Could not create output_dir '%s': %s",
                    out_dir_.c_str(), ec.message().c_str());
    } else {
        RCLCPP_INFO(get_logger(), "Output directory: %s", out_dir_.c_str());
    }
    std::filesystem::path seg_dir(out_dir_);
    for (const auto& entry : std::filesystem::directory_iterator(seg_dir)) {
        const auto& path = entry.path();
        auto name = path.filename().string();
        if (std::regex_match(name, std::regex("frame_.*\\.(tiff|csv|png|jpg)"))) {
            std::filesystem::remove(path);
        }
    }
    RCLCPP_INFO(get_logger(),
        "Cleaning: files frame_*.tiff, frame_*_stats.csv y frame_*_color.(png|jpg) removidos en %s",
        seg_dir.c_str());
}
// ================= Segmentation Image Callbacks ============================
void MaskDetectionNode::synchronized_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& segmentation_image_color,
    const sensor_msgs::msg::Image::ConstSharedPtr& segmentation_image_id) {
    // time stamp
    const auto common_stamp = segmentation_image_color->header.stamp;
    // Fix timestamps
    auto seg_id_fixed    = *segmentation_image_id;
    auto seg_color_fixed = *segmentation_image_color;
    if (seg_id_fixed.header.stamp != common_stamp)
        seg_id_fixed.header.stamp = common_stamp;
    // Publish the fixed messages
    seg_image_color_pub_->publish(seg_color_fixed);
    seg_image_id_pub_->publish(seg_id_fixed);
    // Save the last images
    last_seg_image_color_ = std::make_shared<sensor_msgs::msg::Image>(std::move(seg_color_fixed));
    last_seg_image_id_    = std::make_shared<sensor_msgs::msg::Image>(std::move(seg_id_fixed));
    try_build_legend_and_save_pair();

}

void MaskDetectionNode::try_build_legend_and_save_pair() {
    if (!last_seg_image_color_ || !last_seg_image_id_) {
        return;
    }
    auto color = cv_bridge::toCvShare(last_seg_image_color_, "bgr8")->image;
    auto id_map = cv_bridge::toCvShare(last_seg_image_id_)->image;
    if (color.size() != id_map.size()) {
        RCLCPP_WARN(get_logger(), "Color and ID image sizes do not match.");
        return;
    }
    // Create a legend id -> color
    std::unordered_map<uint16_t, std::array<uint64_t, 4>> legend;
    legend.reserve(512); // Reserve space for 512 unique IDs
    // y axis
    for (int y = 0; y < id_map.rows; ++y) {
        const uint16_t* id_p = id_map.ptr<uint16_t>(y);
        const cv::Vec3b* color_p = color.ptr<cv::Vec3b>(y);
        // x axis
        for (int x = 0; x < id_map.cols; ++x){
            uint16_t id = id_p[x];
            auto &l = legend[id];
            l[0] += color_p[x][0]; // B
            l[1] += color_p[x][1]; // G
            l[2] += color_p[x][2]; // R
            l[3] += 1;             // count
        }
    }
    for (auto key_value: legend) {
        if (key_value.second[3] == 0) continue;
        id_to_bgr_[key_value.first] = cv::Vec3b(
            uint8_t(key_value.second[0] / key_value.second[3]),
            uint8_t(key_value.second[1] / key_value.second[3]),
            uint8_t(key_value.second[2] / key_value.second[3]));
    }
    // Coordinate for saving
    if (seg_image_counter_ < seg_image_limit_){
        const auto& header = last_seg_image_id_->header;
        const int64_t sec  = rclcpp::Time(header.stamp).seconds();
        const uint32_t nsec = rclcpp::Time(header.stamp).nanoseconds() % 1000000000ULL;

        cv::Mat id_cv = cv_bridge::toCvShare(last_seg_image_id_)->image;

        char fname_ids[256];
        snprintf(fname_ids, sizeof(fname_ids),
                "frame_%06d_%ld_%09u_ids.tiff",
                seg_image_counter_, (long)sec, nsec);
        const auto path_ids = (out_dir_ / fname_ids).string();
        cv::imwrite(path_ids, id_cv);
        cv::Mat color_cv = cv_bridge::toCvShare(last_seg_image_color_, "bgr8")->image;
        char fname_color[256];
        snprintf(fname_color, sizeof(fname_color),
                "frame_%06d_%ld_%09u_color.png",
                seg_image_counter_, (long)sec, nsec);
        const auto path_color = (out_dir_ / fname_color).string();
        if (!cv::imwrite(path_color, color_cv)) {
            RCLCPP_WARN(get_logger(), "No se pudo guardar la imagen de color en %s", path_color.c_str());
        }
        const auto legend_path = (out_dir_ / "legend.csv").string();
        save_legend_csv(legend_path, sec, nsec);
        char fname_stats_fmt[256];
        snprintf(fname_stats_fmt, sizeof(fname_stats_fmt),
                "%s", (out_dir_ / "frame_%06d_stats.csv").string().c_str());
        save_frame_stats(fname_stats_fmt, seg_image_counter_, id_cv, sec, nsec);

        seg_image_counter_++;
    }
}

void MaskDetectionNode::save_legend_csv(const std::string& path,
                                        int64_t sec, uint32_t nsec) {
    std::ofstream f(path);
    // CSV file: timestamp_sec, timestamp_nsec, id, blue, green, red, hexadecimal
    f << "timestamp_sec,timestamp_nsec,id,b,g,r,hex\n";
    for (auto &kv : id_to_bgr_) {
        const auto bgr = kv.second;
        char hex[8]; snprintf(hex, sizeof(hex), "#%02X%02X%02X", bgr[2], bgr[1], bgr[0]);
        f << sec << "," << nsec << ","
        << kv.first << "," << int(bgr[0]) << "," << int(bgr[1]) << "," << int(bgr[2])
        << "," << hex << "\n";
    }
    f.close();
}

void MaskDetectionNode::save_frame_stats(const std::string& fmtpath, int idx,
                                         const cv::Mat& ids,
                                         int64_t sec, uint32_t nsec) {
    char buf[256]; snprintf(buf, sizeof(buf), fmtpath.c_str(), idx);
    std::unordered_map<uint16_t, uint32_t> count;
    count.reserve(512);
    for (int y=0; y<ids.rows; ++y) {
        const uint16_t* p = ids.ptr<uint16_t>(y);
        for (int x=0; x<ids.cols; ++x) count[p[x]]++;
    }
    std::ofstream f(buf);
    f << "frame_idx,timestamp_sec,timestamp_nsec,id,pixels\n";
    for (auto &kv : count)
        f << idx << "," << sec << "," << nsec << "," << kv.first << "," << kv.second << "\n";
    f.close();
}

// =========================== ============ ============================
RCLCPP_COMPONENTS_REGISTER_NODE(MaskDetectionNode)
