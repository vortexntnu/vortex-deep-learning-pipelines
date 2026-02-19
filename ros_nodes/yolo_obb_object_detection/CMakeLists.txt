cmake_minimum_required(VERSION 3.8)
project(yolo_obb_object_detection)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY
  launch
  config
  model
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  ${PROJECT_NAME}/${PROJECT_NAME}_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
