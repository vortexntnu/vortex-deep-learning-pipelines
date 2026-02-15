# Sonar Camera Alignment

ROS 2 package that projects camera-based segmentation labels onto the sonar image plane using depth data and TF transforms. Designed for use with the [Stonefish](https://stonefish.readthedocs.io/) underwater simulator.

## Overview

This package bridges the gap between camera-based perception (segmentation) and forward-looking sonar (FLS) imagery. It takes synchronized segmentation and depth images, back-projects each depth pixel into 3D, transforms the point into the sonar frame via TF, and maps the corresponding segmentation label onto a fan-shaped sonar image. The result is a per-pixel label overlay on the sonar display.

### Pipeline

```
Segmentation Image ─┐
                     ├─► seg_depth_sync ──► SegDepthPacket ─┐
Depth Image ─────────┘                                      │
                                                            ├─► sonar_camera_alignment ──► projected_seg ─┐
Sonar Fan Image ────────────────────────────────────────────┘                                             │
                                                                                                         ├─► sonar_projected_overlay ──► overlay
Sonar Fan Image ─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Nodes

### 1. `seg_depth_sync_node`

Synchronizes segmentation and depth image streams (using `message_filters::ApproximateTime`) and bundles them into a single `vortex_msgs/msg/SegDepthPacket` message, which includes both images, their camera intrinsics, and a flat depth array in meters.

| Parameter | Default | Description |
|---|---|---|
| `seg_topic` | `/front_camera_seg/image_raw` | Segmentation image topic |
| `depth_topic` | `/depth_camera/image_depth` | Depth image topic |
| `seg_camera_info_topic` | `/front_camera_seg/camera_info` | Segmentation camera intrinsics |
| `depth_camera_info_topic` | `/depth_camera/camera_info` | Depth camera intrinsics |
| `out_topic` | `/synced/seg_depth_packet` | Output packet topic |
| `sync_queue_size` | `400` | Approximate time sync queue size |
| `max_sync_interval_s` | `5.0` | Maximum allowed timestamp difference (s) |
| `max_depth_m` | `50.0` | Depth values beyond this are set to NaN |
| `depth_scale` | `0.001` | Scale factor for 16UC1 depth (mm to m) |

### 2. `sonar_camera_alignment_node`

Core projection node. For each valid depth pixel it:
1. Back-projects the pixel into the **depth camera frame** using depth intrinsics.
2. Transforms the 3D point into the **segmentation camera frame** via TF, then projects onto the seg image to sample the label.
3. Transforms the same 3D point into the **sonar frame** via TF and maps it to polar (range, bearing) coordinates on the output fan image.

Optionally refines the projection with the sonar acoustic intensity (Otsu or fixed threshold) and a fan-valid mask.

| Parameter | Default | Description |
|---|---|---|
| `packet_topic` | `/synced/seg_depth_packet` | Input `SegDepthPacket` topic |
| `sonar_topic` | `/front_sonar/display_mono` | Sonar fan image topic |
| `sonar_info_topic` | `/front_sonar/sonar_info` | `SonarInfo` metadata topic |
| `projection_mode` | `fan` | Projection mode (`fan` supported) |
| `horizontal_fov_rad` | `2.269` (130 deg) | Horizontal FOV of the sonar in radians |
| `invert_range_axis` | `true` | Flip range axis (far at top) |
| `invert_angle_axis` | `false` | Flip bearing axis |
| `sample_step` | `2` | Depth pixel iteration step (1 = every pixel) |
| `camera_frame_override` | `""` | Override seg camera frame ID |
| `depth_frame_override` | `""` | Override depth camera frame ID |
| `sonar_frame_override` | `""` | Override sonar frame ID |
| `out_width` / `out_height` | `-1` | Force output image size (-1 = use sonar image size) |
| `use_vertical_fov_filter` | `true` | Discard points outside sonar vertical FOV |
| `filter_to_single_label` | `true` | Only project a specific label ID |
| `target_label_id` | `7` | Label ID to project when filtering |
| `refine_with_sonar_intensity` | `true` | Mask projection with acoustic returns |
| `use_otsu_threshold` | `true` | Use Otsu for acoustic thresholding |
| `intensity_threshold` | `40` | Fixed threshold (used when Otsu is off) |
| `acoustic_open_iter` | `1` | Morphological open iterations on acoustic mask |
| `acoustic_close_iter` | `2` | Morphological close iterations on acoustic mask |
| `use_fan_valid_mask` | `true` | Zero out pixels outside the sonar fan region |
| `fan_valid_threshold` | `5` | Intensity threshold to define valid fan pixels |

### 3. `sonar_projected_overlay_node`

Blends the projected segmentation onto the sonar fan image for visualization. Supports per-label colorization (deterministic hue via the golden angle) or a simple red mask overlay.

| Parameter | Default | Description |
|---|---|---|
| `sonar_topic` | `/front_sonar/display_mono` | Sonar fan image topic (base layer) |
| `projected_topic` | `/front_sonar/projected_seg` | Projected segmentation topic |
| `overlay_topic` | `/front_sonar/overlay` | Output overlay topic |
| `alpha` | `0.65` | Blend weight for the overlay |
| `dilate_iter` | `1` | Dilation iterations on the label mask |
| `colorize_labels` | `false` | Color each label ID uniquely |
| `mask_to_sonar_fan` | `true` | Restrict overlay to the visible fan region |
| `fan_mask_threshold` | `5` | Intensity threshold for the fan mask |
| `resize_projected_to_sonar` | `false` | Resize projected image to match sonar (debug) |

### 4. `seg_label_legend_node`

Utility node that inspects every incoming `SegDepthPacket`, extracts the set of unique segmentation label IDs present in the image, logs them, and periodically writes a JSON file mapping IDs to human-readable names.

| Parameter | Default | Description |
|---|---|---|
| `packet_topic` | `/synced/seg_depth_packet` | Input packet topic |
| `ignore_zero_label` | `true` | Exclude label 0 (background) |
| `label_names` | `[]` | Vector of label name strings (index = ID) |
| `json_output_path` | `~/seg_label_legend.json` | Path to write the JSON legend |
| `log_period_ms` | `1000` | Minimum interval between log prints |
| `write_period_ms` | `1000` | Minimum interval between JSON writes |

## Scripts

### `tf_tuner.py`

A PyQt5 GUI tool for interactively tuning a static TF between two frames (by default `camera_rig/Dcam` -> `camera_rig/segmentation_camera_front`). It spawns and respawns `static_transform_publisher` with the slider values. Useful for calibrating the extrinsic transform between depth and segmentation cameras in simulation.

## Dependencies

- **ROS 2** (tested on Humble / Iron)
- `rclcpp`, `sensor_msgs`, `geometry_msgs`
- `tf2`, `tf2_ros`, `tf2_geometry_msgs`
- `message_filters`, `image_transport`, `cv_bridge`
- `OpenCV`
- `stonefish_ros2` (for `SonarInfo` message)
- `vortex_msgs` (for `SegDepthPacket` message)
- `PyQt5` (only for `tf_tuner.py`)

## Building

```bash
cd ~/stonefish_ws
colcon build --packages-select sonar_camera_alignment
source install/setup.bash
```

## Usage

Launch the full pipeline (sync + projection + overlay + legend):

```bash
ros2 launch sonar_camera_alignment sonar_camera_alignment.launch.py
```

Override the parameter file or sonar topic:

```bash
ros2 launch sonar_camera_alignment sonar_camera_alignment.launch.py \
    params_file:=/path/to/custom_params.yaml \
    sonar_topic:=/front_sonar/display_mono
```

## File Structure

```
sonar_camera_alignment/
├── CMakeLists.txt
├── package.xml
├── README.md
├── config/
│   └── sonar_camera_alignment.yaml      # Default parameters
├── include/
│   ├── sonar_camera_alignment.hpp       # Projector node header
│   ├── sonar_projected_overlay.hpp      # Overlay node header
│   ├── seg_depth_sync.hpp               # Sync node header
│   └── seg_label_legend.hpp             # Legend node header
├── launch/
│   ├── sonar_camera_alignment.launch.py # Main launch file
│   ├── static_tf.launch.py              # Static TF: seg camera -> sonar
│   └── depth_to_seg_tf.launch.py        # Static TF: depth camera -> seg camera
├── scripts/
│   └── tf_tuner.py                      # Interactive TF tuning GUI
└── src/
    ├── sonar_camera_alignment_node.cpp  # Projector node
    ├── sonar_projected_overlay_node.cpp # Overlay node
    ├── seg_depth_sync_node.cpp          # Sync node
    └── seg_label_legend_node.cpp        # Legend node
```
