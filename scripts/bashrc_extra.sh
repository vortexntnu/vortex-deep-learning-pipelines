#!/bin/bash
# /ros2_ws/scripts/bashrc_extra.sh for ROS 2 container

# Print actions for clarity
if [ -f /opt/ros/humble/setup.bash ]; then
    echo "[bashrc_extra] Sourcing ROS 2 underlay: /opt/ros/humble/setup.bash"
    source /opt/ros/humble/setup.bash
else
    echo "[bashrc_extra] ROS 2 underlay not found: /opt/ros/humble/setup.bash"
fi

if [ -f /ros2_ws/install/setup.bash ]; then
    echo "[bashrc_extra] Sourcing workspace overlay: /ros2_ws/install/setup.bash"
    source /ros2_ws/install/setup.bash
else
    echo "[bashrc_extra] Workspace overlay not found: /ros2_ws/install/setup.bash"
fi

# Add your custom shell commands, aliases, or environment variables below
