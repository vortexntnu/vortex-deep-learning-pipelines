# mask_classifier

A ROS 2 package for automatic classification of segmented objects in simulation or real environments.  
It subscribes to segmentation image topics and processes masks to extract class information, generate detections, and publish bounding boxes.

---

## Overview

`mask_classifier` is designed to work with segmentation images (color and ID) produced by a segmentation camera, such as the one in the [Stonefish Simulator](https://pure.hw.ac.uk/ws/portalfiles/portal/146711343/2502.11887v1.pdf).  
It classifies objects based on pixel-wise segmentation labels and exports annotated detections for downstream perception modules (e.g., YOLO training, or UNet segmentation).

Main features:
- Subscribes to segmentation topics (`/segmentation_image_color`, `/segmentation_image_id`).
- Uses a legend CSV file to map class IDs to names and colors.
- Filters out small regions below a configurable pixel threshold.
- Supports synchronized image and ID inputs.

---

## Package Structure

```

mask_classifier/
├── CMakeLists.txt
├── package.xml
├── launch/
│   └── mask_classifier.launch.py
├── mask_classifier/
│   ├── __init__.py
│   └── node.py
├── resource/
│   └── mask_classifier
├── scripts/
│   └── live_mask_viewer.py
├── test/
└── LICENSE

````

---

## Dependencies

Ensure the following are installed in your ROS 2 workspace:

- **ROS 2 Humbles**
- `rclpy`
- `sensor_msgs`, `vision_msgs`, `std_msgs`
- `cv_bridge`
- `opencv-python`
- `numpy`

Install Python dependencies if needed:
```bash
pip install opencv-python numpy
````

---

## Building

Clone the package inside your ROS 2 workspace:

```bash
cd ~/stonefish_ws/src
git clone https://github.com/vortexntnu/vortex-deep-learning-pipelines
```

Then build and source it:

```bash
cd ~/stonefish_ws
colcon build --packages-select mask_classifier --symlink-install
source install/setup.bash
```

---

## Running

Launch the node using the provided launch file:

```bash
ros2 launch mask_classifier mask_classifier.launch.py
```

### Parameters

| Name                                 | Type     | Default                     | Description                                    |
| ------------------------------------ | -------- | --------------------------- | ---------------------------------------------- |
| `segmentation_image_color_sub_topic` | `string` | `/segmentation_image_color` | RGB segmentation input topic                   |
| `segmentation_image_id_sub_topic`    | `string` | `/segmentation_image_id`    | Class-ID segmentation topic                    |
| `legend_csv_path`                    | `string` | `""`                        | Path to CSV mapping of ID → class name / color |
| `min_pixels_per_object`              | `int`    | `200`                       | Minimum pixel count to consider a detection    |

---

## Node Description

### `mask_classifier.node.MaskClassifierNode`

* Subscribes to segmentation image topics.
* Converts images via `cv_bridge`.
* Parses the legend CSV file to match ID → label.
* Computes bounding boxes from mask regions.

---
## live_mask_viewer.py
An interface that outstands the object selected from other objects. It uses the `id_label_map.json`.

---

## 📂 Example Use Case

Used within the Vortex or Stonefish simulation pipeline to:

* Automatically label dataset frames.
* Generate bounding boxes for YOLOv8.
* Export pixel masks for UNet semantic segmentation.
* Modify the ids in `id_label_map.json`, which is generated in `$HOME/seg_frames`.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

Developed by **Liangji Zhu**

Part of the **Vortex Deep Learning Pipelines** project for perception model automation.
