# yolo_inference
Real-time object detection using YOLOv8. This node subscribes to an image topic, runs inference, and publishes both structured detections and an annotated image with bounding boxes.

## Run
```bash
ros2 launch yolo_inference yolo_inference.launch.py
```
You can also select the compute device with the `device` argument:
```bash
# Force inference on CPU
ros2 launch yolo_inference yolo_inference.launch.py device:=cpu

# Use GPU 0
ros2 launch yolo_inference yolo_inference.launch.py device:=0
```
**By default, the node runs on GPU.**


## Topics
### Subscribed
- `sensor_msgs/Image` - Input color image,
### Published
- `/yolo/detections` -> `vision_msgs/Detection2DArray`
- `/yolo/annotated` -> `sensor_msgs/Image`
