# yolo_obb_object_detection
Real-time object detection using YOLOv26. This node subscribes to an image topic, runs inference, and publishes both structured detections and an annotated image with oriented bounding boxes (OBB).

## Run
```bash
ros2 launch yolo_obb_object_detection yolo_obb_object_detection.launch.py
```
You can also select the compute device with the `device` argument:
```bash
# Run inference on CPU
ros2 launch yolo_obb_object_detection yolo_obb_object_detection.launch.py device:=cpu

# Run inference on GPU 0
ros2 launch yolo_obb_object_detection yolo_obb_object_detection.launch.py device:=0
```
**By default, the node runs on CPU.**

## Topics
### Subscribed
- `sensor_msgs/Image`
### Published
- `/yolo_obb_object_detection/detections` -> `vision_msgs/Detection2DArray`
- `/yolo_obb_object_detection/annotated` -> `sensor_msgs/Image`
