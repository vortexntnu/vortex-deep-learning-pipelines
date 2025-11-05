# Mask Detection ROS 2 Package

## Overview

**mask-detection** is a ROS 2 package designed to process segmentation camera images.

The package synchronizes segmentation color images and ID images and identifies objects based on color–ID mapping.

---

## Features

- Subscribes to segmentation color and ID image topics.
- Performs mask-based object detection from color–ID pairs.
- Configurable pixel thresholds to filter small objects.
- Designed for integration with **YOLOv8 / UNet dataset generation pipelines**.
- Modular C++ implementation with ROS 2 launch and YAML configuration.

---

## Structure

```

mask-detection/
├── include/mask_detection/mask_detection.hpp     # Main C++ header
├── src/mask_detection.cpp                        # Core node implementation
├── config/mask_detection_params.yaml             # Node parameters
├── launch/                                       # Launch files for running the node
├── CMakeLists.txt                                # Build configuration
├── package.xml                                   # ROS 2 package metadata
└── README.md                                     # Documentation

````

---

## Building

Build the package with `colcon`:

```bash
cd ~/stonefish_ws
colcon build --packages-up-to mask-detection --symlink-install
source install/setup.bash
```

---

## Usage

You can launch the node with a ROS 2 launch file, e.g.:

```bash
ros2 launch mask_detection mask_detection.launch.py
```
---

## Configuration

Parameters are defined in `config/mask_detection_params.yaml`.
Example:

```yaml
    # Orca
    # segmentation_image_color_sub_topic: "/cam_segmentation/image_color"
    # segmentation_image_id_sub_topic:    "/cam_segmentation/image_raw"
    # camera_rig
    segmentation_image_color_sub_topic: "/front_camera_seg/image_color"
    segmentation_image_id_sub_topic:    "/front_camera_seg/image_raw"
```

| Parameter                            | Type   | Description                              |
| ------------------------------------ | ------ | ---------------------------------------- |
| `segmentation_image_color_sub_topic` | string | Topic name for segmentation color image. |
| `segmentation_image_id_sub_topic`    | string | Topic name for segmentation ID image.    |
---

## Published Topics

| Topic                       | Type                | Description                  |
| --------------------------- | ------------------- | ---------------------------- |
| `/segmentation_image_color` | `sensor_msgs/Image` | RGB segmentation image.      |
| `/segmentation_image_id`    | `sensor_msgs/Image` | ID-coded segmentation image. |

---

## Subscribed Topics

| Topic                       | Type                | Description                  |
| --------------------------- | ------------------- | ---------------------------- |
| `/segmentation_image_color` | `sensor_msgs/Image` | RGB segmentation image.      |
| `/segmentation_image_id`    | `sensor_msgs/Image` | ID-coded segmentation image. |

---

## Integration

This package can be integrated with:

* **`mask_classifier`** or YOLO-based networks for training dataset generation.
* The **Stonefish simulator’s segmentation camera** for synthetic data collection.

---

## Example Workflow

1. Run the Stonefish simulation with a segmentation camera enabled.
2. Launch `mask-detection` to extract labeled object detections.
3. Optionally record outputs with `mask_classifier`.
4. Use generated masks and detections for model training (e.g., YOLOv8, UNet), it generates in the `$HOME\seg_frames` directory. It also generates a `legend.csv`.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

Developed by **Liangji Zhu**

Part of the **Vortex Deep Learning Pipelines** project for perception model automation.

