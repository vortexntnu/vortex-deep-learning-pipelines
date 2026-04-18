# yolo_classify
Real-time image classification using YOLO. This node subscribes to an image topic and runs inference, logging the top-1 predicted class and confidence.

## Run
```bash
ros2 launch yolo_classify classifier_node.launch.py
```
You can also select the compute device with the `device` argument:
```bash
# Run inference on CPU
ros2 launch yolo_classify classifier_node.launch.py device:=cpu

# Run inference on GPU 0
ros2 launch yolo_classify classifier_node.launch.py device:=0
```
**By default, the node runs on CPU.**

## Topics
### Subscribed
- `sensor_msgs/Image`
