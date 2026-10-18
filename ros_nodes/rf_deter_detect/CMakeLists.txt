cmake_minimum_required(VERSION 3.8)
project(rf_deter_detect)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY
  launch
  config
  model
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(PROGRAMS
  ${PROJECT_NAME}/rf_deter_detection_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
