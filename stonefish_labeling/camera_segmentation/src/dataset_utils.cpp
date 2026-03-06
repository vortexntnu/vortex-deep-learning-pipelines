#include "camera_segmentation/dataset_utils.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <regex>
#include <system_error>
#include <vector>

namespace vortex::camera_segmentation::dataset_utils {

bool prepare_output_directory(const std::filesystem::path &out_dir,
                              const rclcpp::Logger &logger) {
  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  if (ec) {
    RCLCPP_WARN(logger, "Could not create output_dir '%s': %s",
                out_dir.c_str(), ec.message().c_str());
    return false;
  }

  RCLCPP_INFO(logger, "Output directory: %s", out_dir.c_str());

  for (const auto &entry : std::filesystem::directory_iterator(out_dir)) {
    const auto name = entry.path().filename().string();
    if (std::regex_match(name, std::regex(R"(frame_.*\.(png|tiff|csv|jpg))")) ||
        name == "legend.csv") {
      std::filesystem::remove(entry.path(), ec);
      if (ec) {
        RCLCPP_WARN(logger, "Could not remove '%s': %s",
                    entry.path().c_str(), ec.message().c_str());
      }
    }
  }

  return true;
}

uint32_t pack_rgb(const cv::Vec3b &bgr) {
  const uint32_t r = static_cast<uint32_t>(bgr[2]);
  const uint32_t g = static_cast<uint32_t>(bgr[1]);
  const uint32_t b = static_cast<uint32_t>(bgr[0]);
  return (r << 16) | (g << 8) | b;
}

bool save_legend_csv(const std::filesystem::path &out_dir,
                     const std::unordered_map<uint32_t, uint16_t> &rgb_to_id,
                     const rclcpp::Logger &logger) {
  const auto legend_path = (out_dir / "legend.csv").string();
  std::ofstream f(legend_path);
  if (!f.is_open()) {
    RCLCPP_WARN(logger, "Could not open legend file '%s' for writing.",
                legend_path.c_str());
    return false;
  }

  f << "id,r,g,b,hex\n";

  std::vector<std::pair<uint16_t, uint32_t>> ordered;
  ordered.reserve(rgb_to_id.size());

  for (const auto &kv : rgb_to_id) {
    ordered.emplace_back(kv.second, kv.first);
  }

  std::sort(ordered.begin(), ordered.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });

  for (const auto &kv : ordered) {
    const uint16_t id = kv.first;
    const uint32_t rgb = kv.second;

    const uint8_t r = static_cast<uint8_t>((rgb >> 16) & 0xFF);
    const uint8_t g = static_cast<uint8_t>((rgb >> 8) & 0xFF);
    const uint8_t b = static_cast<uint8_t>(rgb & 0xFF);

    char hex[8];
    std::snprintf(hex, sizeof(hex), "#%02X%02X%02X", r, g, b);

    f << id << "," << static_cast<int>(r) << "," << static_cast<int>(g) << ","
      << static_cast<int>(b) << "," << hex << "\n";
  }

  return true;
}

bool save_frame_pair(const std::filesystem::path &out_dir,
                     int frame_counter,
                     const cv::Mat &front_color,
                     const cv::Mat &id_mask,
                     const rclcpp::Logger &logger,
                     std::string *image_path_out,
                     std::string *mask_path_out) {
  char base_name[64];
  std::snprintf(base_name, sizeof(base_name), "frame_%06d", frame_counter);

  const auto image_path =
      (out_dir / (std::string(base_name) + ".png")).string();
  const auto mask_path =
      (out_dir / (std::string(base_name) + "_mask.tiff")).string();

  if (!cv::imwrite(image_path, front_color)) {
    RCLCPP_WARN(logger, "Could not save image to %s", image_path.c_str());
    return false;
  }

  if (!cv::imwrite(mask_path, id_mask)) {
    RCLCPP_WARN(logger, "Could not save mask to %s", mask_path.c_str());
    return false;
  }

  if (image_path_out != nullptr) {
    *image_path_out = image_path;
  }
  if (mask_path_out != nullptr) {
    *mask_path_out = mask_path;
  }

  return true;
}

}  // namespace vortex::camera_segmentation::dataset_utils
