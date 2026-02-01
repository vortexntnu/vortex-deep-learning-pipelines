#ifndef MASK_DETECTION_HPP
#define MASK_DETECTION_HPP

#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <filesystem>

#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/opencv.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
const int SEG_IMAGE_LIMIT = 1000000; // -1 for unlimited


class MaskDetectionNode : public rclcpp::Node {
public:
    explicit MaskDetectionNode(const rclcpp::NodeOptions& options);
    ~MaskDetectionNode() {};

private:
    std::filesystem::path out_dir_;
    bool sync_with_front_camera_ = true;
    int sync_queue_size_ = 20;
    double sync_max_interval_sec_ = 0.5;
    // Segmentation camera to obtain mask
    message_filters::Subscriber<sensor_msgs::msg::Image> segmentation_image_color_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> segmentation_image_id_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> front_camera_color_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr front_camera_color_sub_direct_;
    std::mutex last_front_mutex_;

    // Synchronization policy
    using SyncPolicy3 = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,                       // Segmentation image color
        sensor_msgs::msg::Image,                       // Segmentation image id
        sensor_msgs::msg::Image>;                      // Front camera color
    using SyncPolicy2 = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,                       // Segmentation image color
        sensor_msgs::msg::Image>;                      // Segmentation image id
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy3>> sync3_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy2>> sync2_;

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr seg_image_color_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr seg_image_id_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr front_camera_color_pub_;

    // Frames for segmentation
    int seg_image_counter_ = 0;        
    const int seg_image_limit_ = SEG_IMAGE_LIMIT;
    sensor_msgs::msg::Image::SharedPtr last_seg_image_color_;
    sensor_msgs::msg::Image::SharedPtr last_seg_image_id_;
    sensor_msgs::msg::Image::SharedPtr last_front_camera_color_;
    std::unordered_map<uint16_t, cv::Vec3b> id_to_bgr_;
    /**
     * @brief Callback function for synchronized the color and id of
     * segmentation images.
     *
     * This function is triggered when synchronized messages for a depth image,
     * a color image, and a segmentation images are received.
     */
     void synchronized_callback3(
        const sensor_msgs::msg::Image::ConstSharedPtr& segmentation_image_color,
        const sensor_msgs::msg::Image::ConstSharedPtr& segmentation_image_id,
        const sensor_msgs::msg::Image::ConstSharedPtr& front_camera_color);
     void synchronized_callback2(
        const sensor_msgs::msg::Image::ConstSharedPtr& segmentation_image_color,
        const sensor_msgs::msg::Image::ConstSharedPtr& segmentation_image_id);
     void front_camera_callback(const sensor_msgs::msg::Image::ConstSharedPtr& front_camera_color);
     void handle_triplet_and_save(
        const sensor_msgs::msg::Image::ConstSharedPtr& segmentation_image_color,
        const sensor_msgs::msg::Image::ConstSharedPtr& segmentation_image_id,
        const sensor_msgs::msg::Image::ConstSharedPtr& front_camera_color);
    
    void try_build_legend_and_save_pair();
    void save_legend_csv(const std::string& path, int64_t sec, uint32_t nsec);
    void save_frame_stats(const std::string& fmtpath, int idx, const cv::Mat& ids,
                          int64_t sec, uint32_t nsec);
};

#endif // MASK_DETECTION_HPP