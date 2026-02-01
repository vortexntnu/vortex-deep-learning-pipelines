#include "mask_detection/mask_detection.hpp"

// For binding synchronized callback
using std::placeholders::_1;    // segmentation image color
using std::placeholders::_2;    // segmentation image id
using std::placeholders::_3;    // front camera color

MaskDetectionNode::MaskDetectionNode(const rclcpp::NodeOptions& options)
    : Node("mask_detection_node", options) {
    // Topics to subscribe to
    auto segmentation_image_color_sub_topic =
        this->declare_parameter<std::string>("segmentation_image_color_sub_topic", "/front_camera_seg/image_color");
    auto segmentation_image_id_sub_topic =
        this->declare_parameter<std::string>("segmentation_image_id_sub_topic", "/front_camera_seg/image_raw");
    auto front_camera_color_sub_topic =
        this->declare_parameter<std::string>("front_camera_color_sub_topic", "/front_camera/image_color");

    // Sync behavior (common reason for "no output" if stamps don't align)
    sync_with_front_camera_ = this->declare_parameter<bool>("sync_with_front_camera", true);
    sync_queue_size_ = this->declare_parameter<int>("sync_queue_size", 20);
    sync_max_interval_sec_ = this->declare_parameter<double>("sync_max_interval_sec", 0.5);
    if (sync_queue_size_ < 1) sync_queue_size_ = 1;
    if (sync_max_interval_sec_ <= 0.0) sync_max_interval_sec_ = 0.5;

    // Quality of Service settings
    auto qos = rclcpp::SensorDataQoS();  // Camera data QoS
    segmentation_image_color_sub_.subscribe(this, segmentation_image_color_sub_topic, qos.get_rmw_qos_profile());
    segmentation_image_id_sub_.subscribe(this, segmentation_image_id_sub_topic, qos.get_rmw_qos_profile());

    if (sync_with_front_camera_) {
        front_camera_color_sub_.subscribe(this, front_camera_color_sub_topic, qos.get_rmw_qos_profile());
        // 3-way synchronization (seg_color, seg_id, front_color)
        sync3_ = std::make_shared<message_filters::Synchronizer<SyncPolicy3>>(
            SyncPolicy3(sync_queue_size_),
            segmentation_image_color_sub_,
            segmentation_image_id_sub_,
            front_camera_color_sub_);
        sync3_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(sync_max_interval_sec_));
        sync3_->registerCallback(std::bind(&MaskDetectionNode::synchronized_callback3, this, _1, _2, _3));
        RCLCPP_INFO(get_logger(),
            "Sync mode: 3-way (with front camera). queue=%d max_interval=%.3fs",
            sync_queue_size_, sync_max_interval_sec_);
    } else {
        // 2-way sync (seg_color, seg_id) + latest front camera (no strict stamp matching)
        front_camera_color_sub_direct_ = this->create_subscription<sensor_msgs::msg::Image>(
            front_camera_color_sub_topic,
            qos,
            std::bind(&MaskDetectionNode::front_camera_callback, this, std::placeholders::_1));

        sync2_ = std::make_shared<message_filters::Synchronizer<SyncPolicy2>>(
            SyncPolicy2(sync_queue_size_),
            segmentation_image_color_sub_,
            segmentation_image_id_sub_);
        sync2_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(sync_max_interval_sec_));
        sync2_->registerCallback(std::bind(&MaskDetectionNode::synchronized_callback2, this, _1, _2));
        RCLCPP_INFO(get_logger(),
            "Sync mode: 2-way (segmentation only) + latest front camera. queue=%d max_interval=%.3fs",
            sync_queue_size_, sync_max_interval_sec_);
    }

    // Publishers
    seg_image_color_pub_ =
        this->create_publisher<sensor_msgs::msg::Image>("segmentation_image_color", 10);
    seg_image_id_pub_ =
        this->create_publisher<sensor_msgs::msg::Image>("segmentation_image_id", 10);
    front_camera_color_pub_ = 
        this->create_publisher<sensor_msgs::msg::Image>("front_camera_image_color_synced", 10);

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
        "Cleaning: files frame_*.tiff, frame_*_stats.csv and frame_*_color.(png|jpg) removed in %s",
        seg_dir.c_str());
}
// ================= Segmentation Image Callbacks ============================
void MaskDetectionNode::synchronized_callback3(
    const sensor_msgs::msg::Image::ConstSharedPtr& segmentation_image_color,
    const sensor_msgs::msg::Image::ConstSharedPtr& segmentation_image_id,
    const sensor_msgs::msg::Image::ConstSharedPtr& front_camera_color) {
    RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 2000,
        "SYNC3 fired at %u.%u",
        segmentation_image_color->header.stamp.sec,
        segmentation_image_color->header.stamp.nanosec);
    handle_triplet_and_save(segmentation_image_color, segmentation_image_id, front_camera_color);
}

void MaskDetectionNode::front_camera_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& front_camera_color) {
    std::lock_guard<std::mutex> lk(last_front_mutex_);
    last_front_camera_color_ = std::make_shared<sensor_msgs::msg::Image>(*front_camera_color);
}

void MaskDetectionNode::synchronized_callback2(
    const sensor_msgs::msg::Image::ConstSharedPtr& segmentation_image_color,
    const sensor_msgs::msg::Image::ConstSharedPtr& segmentation_image_id) {
    sensor_msgs::msg::Image::SharedPtr front;
    {
        std::lock_guard<std::mutex> lk(last_front_mutex_);
        front = last_front_camera_color_;
    }
    if (!front) {
        RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000,
            "SYNC2 fired but no front camera image received yet; skipping save.");
        return;
    }
    RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 2000,
        "SYNC2 fired at %u.%u (using latest front camera)",
        segmentation_image_color->header.stamp.sec,
        segmentation_image_color->header.stamp.nanosec);
    handle_triplet_and_save(segmentation_image_color, segmentation_image_id, front);
}

void MaskDetectionNode::handle_triplet_and_save(
    const sensor_msgs::msg::Image::ConstSharedPtr& segmentation_image_color,
    const sensor_msgs::msg::Image::ConstSharedPtr& segmentation_image_id,
    const sensor_msgs::msg::Image::ConstSharedPtr& front_camera_color) {
    // Use segmentation_color stamp as canonical stamp
    const auto common_stamp = segmentation_image_color->header.stamp;

    // Fix timestamps
    auto seg_id_fixed       = *segmentation_image_id;
    auto seg_color_fixed    = *segmentation_image_color;
    auto front_color_fixed  = *front_camera_color;

    if (seg_id_fixed.header.stamp != common_stamp) seg_id_fixed.header.stamp = common_stamp;
    if (front_color_fixed.header.stamp != common_stamp) front_color_fixed.header.stamp = common_stamp;

    // Publish the fixed messages
    seg_image_color_pub_->publish(seg_color_fixed);
    seg_image_id_pub_->publish(seg_id_fixed);
    front_camera_color_pub_->publish(front_color_fixed);

    // Save the last images
    last_seg_image_color_ = std::make_shared<sensor_msgs::msg::Image>(std::move(seg_color_fixed));
    last_seg_image_id_    = std::make_shared<sensor_msgs::msg::Image>(std::move(seg_id_fixed));
    {
        std::lock_guard<std::mutex> lk(last_front_mutex_);
        last_front_camera_color_ = std::make_shared<sensor_msgs::msg::Image>(std::move(front_color_fixed));
    }

    try_build_legend_and_save_pair();
}

void MaskDetectionNode::try_build_legend_and_save_pair() {
    sensor_msgs::msg::Image::SharedPtr front_copy;
    {
        std::lock_guard<std::mutex> lk(last_front_mutex_);
        front_copy = last_front_camera_color_;
    }
    if (!last_seg_image_color_ || !last_seg_image_id_ || !front_copy) {
        return;
    }
    cv::Mat color;
    cv::Mat id_map_raw;
    cv::Mat front_color;
    try {
        color       = cv_bridge::toCvShare(last_seg_image_color_, "bgr8")->image;
        id_map_raw  = cv_bridge::toCvShare(last_seg_image_id_)->image;
        front_color = cv_bridge::toCvShare(front_copy, "bgr8")->image;
    } catch (const std::exception& e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000,
            "cv_bridge conversion failed: %s", e.what());
        return;
    }

    // Enforce uint16 ID map (common encodings: 16UC1/mono16; sometimes 32SC1)
    cv::Mat id_map;
    if (id_map_raw.type() == CV_16UC1) {
        id_map = id_map_raw;
    } else {
        id_map_raw.convertTo(id_map, CV_16U);
        RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000,
            "ID map type was %d, converted to CV_16UC1 for processing.", id_map_raw.type());
    }

    if (color.size() != id_map.size()) {
        RCLCPP_WARN(get_logger(), "Color and ID image sizes do not match.");
        return;
    }

    // --------- Build legend ID -> average color BGR ---------
    std::unordered_map<uint16_t, std::array<uint64_t, 4>> legend;
    legend.reserve(512);

    for (int y = 0; y < id_map.rows; ++y) {
        const uint16_t*   id_p    = id_map.ptr<uint16_t>(y);
        const cv::Vec3b*  color_p = color.ptr<cv::Vec3b>(y);
        for (int x = 0; x < id_map.cols; ++x) {
            uint16_t id = id_p[x];
            auto &l = legend[id];
            l[0] += color_p[x][0]; // B
            l[1] += color_p[x][1]; // G
            l[2] += color_p[x][2]; // R
            l[3] += 1;             // count
        }
    }

    for (auto &key_value : legend) {
        if (key_value.second[3] == 0) continue;
        id_to_bgr_[key_value.first] = cv::Vec3b(
            static_cast<uint8_t>(key_value.second[0] / key_value.second[3]),
            static_cast<uint8_t>(key_value.second[1] / key_value.second[3]),
            static_cast<uint8_t>(key_value.second[2] / key_value.second[3])
        );
    }

    // --------- Save images and CSVs ---------
    if (seg_image_counter_ < seg_image_limit_) {
        // Timestamp (we still use it in legend.csv and stats)
        const auto& header  = last_seg_image_id_->header;
        const int64_t  sec  = rclcpp::Time(header.stamp).seconds();
        const uint32_t nsec = rclcpp::Time(header.stamp).nanoseconds() % 1000000000ULL;

        cv::Mat id_cv     = id_map;     // already CV_16UC1
        cv::Mat color_cv  = color;
        cv::Mat front_cv  = front_color;

        // --- SHARED BASE NAME ---
        //   Image:  frame_000000.png
        //   Mask: frame_000000_mask.png
        char base_name[64];
        std::snprintf(base_name, sizeof(base_name), "frame_%06d", seg_image_counter_);

        // ----- IDs uint16 -----
        char fname_ids[256];
        std::snprintf(fname_ids, sizeof(fname_ids), "%s_ids.tiff", base_name);
        const auto path_ids = (out_dir_ / fname_ids).string();
        if (!cv::imwrite(path_ids, id_cv)) {
            RCLCPP_WARN(get_logger(), "Cannot save ID map to %s", path_ids.c_str());
        }

        // ----- Segmentation mask (color) -----
        // This will be the mask you use in Roboflow.
        char fname_color[256];
        std::snprintf(fname_color, sizeof(fname_color), "%s_mask.png", base_name);
        const auto path_color = (out_dir_ / fname_color).string();
        if (!cv::imwrite(path_color, color_cv)) {
            RCLCPP_WARN(get_logger(), "Cannot save color mask to %s", path_color.c_str());
        }

        // ----- Front camera RGB image -----
        // This will be the original image for Roboflow.
        char fname_front[256];
        std::snprintf(fname_front, sizeof(fname_front), "%s.png", base_name);
        const auto path_front = (out_dir_ / fname_front).string();
        if (!cv::imwrite(path_front, front_cv)) {
            RCLCPP_WARN(get_logger(), "Cannot save front image to %s", path_front.c_str());
        }

        // ----- Global color legend -----
        const auto legend_path = (out_dir_ / "legend.csv").string();
        save_legend_csv(legend_path, sec, nsec);

        // ----- Statistics per frame -----
        // Result: frame_000000_stats.csv, etc.
        char fname_stats_fmt[256];
        std::snprintf(fname_stats_fmt, sizeof(fname_stats_fmt),
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
