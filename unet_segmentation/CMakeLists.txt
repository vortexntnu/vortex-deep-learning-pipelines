cmake_minimum_required(VERSION 3.8)
project(unet_segmentation)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY
  launch
  config
  model
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  ${PROJECT_NAME}/interface.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()