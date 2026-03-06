#ifndef CAMERA_SEGMENTATION__UTILS_HPP_
#define CAMERA_SEGMENTATION__UTILS_HPP_

#include <filesystem>
#include <string>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>

namespace vortex::camera_segmentation::dataset_utils {

bool prepare_output_directory(const std::filesystem::path &out_dir,
                              const rclcpp::Logger &logger);

uint32_t pack_rgb(const cv::Vec3b &bgr);

bool save_legend_csv(const std::filesystem::path &out_dir,
                     const std::unordered_map<uint32_t, uint16_t> &rgb_to_id,
                     const rclcpp::Logger &logger);

bool save_frame_pair(const std::filesystem::path &out_dir,
                     int frame_counter,
                     const cv::Mat &front_color,
                     const cv::Mat &id_mask,
                     const rclcpp::Logger &logger,
                     std::string *image_path_out = nullptr,
                     std::string *mask_path_out = nullptr);

}  // namespace vortex::camera_segmentation::dataset_utils

#endif // CAMERA_SEGMENTATION__UTILS_HPP_
