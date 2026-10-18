# rf_deter_detect

ROS 2 object detection using a trained RF-DETR checkpoint. The node subscribes
to a `sensor_msgs/Image`, then publishes both `vision_msgs/Detection2DArray`
detections and an annotated image.

## Model

Copy the checkpoint produced by RF-DETR training into the package model folder:

```bash
cp /path/to/checkpoint_best_total.pth model/checkpoint_best_total.pth
```

Alternatively, set `rf_detr_model` in the parameter file to an absolute path.
RF-DETR infers the Nano/Small/Medium/Large architecture from a checkpoint. An
exported `.onnx` model can also be used directly through ONNX Runtime. Configure
`class_names` in the parameter file in the same order used during training.

## Install and run

Install the Python inference dependency in the environment used by ROS:

```bash
python3 -m pip install -r requirements.txt
```

The requirements use the GPU ONNX Runtime build, which also includes the CPU
provider. Do not install `onnxruntime` and `onnxruntime-gpu` together: both
packages write the same Python module and the CPU package can hide CUDA support.

Build from the ROS workspace, source it, and launch:

```bash
colcon build --packages-select rf_deter_detect --symlink-install
source install/setup.bash
ros2 launch rf_deter_detect rf_deter_detection.launch.py
```

Select a GPU with either `device:=cuda`, `device:=cuda:0`, or the YOLO-style
shortcut `device:=0`:

```bash
ros2 launch rf_deter_detect rf_deter_detection.launch.py device:=0
```

## Topics

- Subscribes to `/nautilus/front_camera/image_color` (`sensor_msgs/Image`)
- Publishes `/rf_deter_detection/detections` (`vision_msgs/Detection2DArray`)
- Publishes `/rf_deter_detection/annotated` (`sensor_msgs/Image`)

Topic names, checkpoint path, confidence threshold, device, and checkpoint
trust behavior are configurable in `config/rf_deter_detection_params.yaml`.

The default model is an FP16 conversion optimized for NVIDIA inference. The
original FP32 ONNX model remains available if maximum numerical parity is more
important than throughput. Camera input uses a depth-one sensor queue so stale
frames are discarded when the camera is faster than inference. Annotated images
are only rendered when that output has a subscriber.
