cmake_minimum_required(VERSION 3.8)
project(sonar_segmentation)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(stonefish_ros2 REQUIRED)
find_package(vortex_utils REQUIRED)
find_package(vortex_msgs REQUIRED)
find_package(stonefish_ros2 REQUIRED)
find_package(vortex_utils_ros REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(sonar_segmentation_node
  src/sonar_segmentation_node.cpp
  src/main.cpp
)

ament_target_dependencies(
  sonar_segmentation_node
  rclcpp
  sensor_msgs
  cv_bridge
  stonefish_ros2
  vortex_utils
  vortex_msgs
  vortex_utils_ros
)

target_link_libraries(
  sonar_segmentation_node
  ${OpenCV_LIBRARIES}
)

install(TARGETS
  sonar_segmentation_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
