cmake_minimum_required(VERSION 3.8)
project(sonar_camera_alignment)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(stonefish_ros2 REQUIRED)
find_package(vortex_msgs REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(message_filters REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(include)

# --------------------------
# Executable 1: projector
# --------------------------
add_executable(sonar_camera_alignment_node
  src/sonar_camera_alignment_node.cpp
)

ament_target_dependencies(sonar_camera_alignment_node
  rclcpp
  sensor_msgs
  geometry_msgs
  stonefish_ros2
  vortex_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  message_filters
  image_transport
  cv_bridge
)

target_link_libraries(sonar_camera_alignment_node
  ${OpenCV_LIBRARIES}
)

# --------------------------
# Executable 2: overlay
# --------------------------
add_executable(sonar_projected_overlay_node
  src/sonar_projected_overlay_node.cpp
)

ament_target_dependencies(sonar_projected_overlay_node
  rclcpp
  sensor_msgs
  message_filters
  image_transport
  cv_bridge
  vortex_msgs 
)

target_link_libraries(sonar_projected_overlay_node
  ${OpenCV_LIBRARIES}
)

# --------------------------
# Executable 3: seg_depth_sync
# --------------------------
add_executable(seg_depth_sync_node
  src/seg_depth_sync_node.cpp
)

ament_target_dependencies(seg_depth_sync_node
  rclcpp
  sensor_msgs
  message_filters
  cv_bridge
  vortex_msgs
)

target_link_libraries(seg_depth_sync_node
  ${OpenCV_LIBRARIES}
)

# --------------------------
# Executable 4: seg_label_legend
# --------------------------
add_executable(seg_label_legend_node
  src/seg_label_legend_node.cpp
)

ament_target_dependencies(seg_label_legend_node
  rclcpp
  sensor_msgs
  cv_bridge
  vortex_msgs
)

target_link_libraries(seg_label_legend_node
  ${OpenCV_LIBRARIES}
)

# --------------------------
# Install
# --------------------------
install(TARGETS
  sonar_camera_alignment_node
  sonar_projected_overlay_node
  seg_depth_sync_node
  seg_label_legend_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

ament_package()
