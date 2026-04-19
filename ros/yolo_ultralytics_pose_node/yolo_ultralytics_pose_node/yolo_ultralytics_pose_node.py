#!/usr/bin/env python3

# --- IMPORTS ---
# These are like Java "import" statements. Each library gives us tools we need.
#
# rclpy  = the ROS2 Python library (think: the ROS2 SDK)
# Node   = the base class every ROS2 node must extend (like extending a Java base class)
# Parameter = used to declare and read config values (like reading application.properties)
# qos_profile_sensor_data = a preset for camera/sensor topics (drops old frames instead of queuing)
# Image (sensor_msgs) = the ROS2 message type that carries a camera frame
# Detection2DArray etc. = ROS2 message types for publishing detection results
# CvBridge = converts between ROS2 Image messages and OpenCV numpy arrays
# YOLO = the Ultralytics model class
# math = standard Python math library (we need atan2 to compute the angle)

import math
import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


# --- THE NODE CLASS ---
# In Java this would be:  public class YoloUltralyticsPoseNode extends RosNode { ... }
# Every ROS2 node is a class that extends Node.
class YoloUltralyticsPoseNode(Node):

    def __init__(self):
        # This is the constructor, called once when the node starts.
        # super().__init__('name') registers this node on the ROS2 network with that name.
        # In Java: super("yolo_ultralytics_pose_node");
        super().__init__('yolo_ultralytics_pose_node')

        # Load config values (model path, topic names, etc.) from the YAML params file.
        self._load_parameters()

        # Load the YOLO model weights file from disk.
        self.model = self._load_model()

        # CvBridge translates between ROS Image messages and OpenCV numpy arrays.
        # You need this every time you want to do computer vision on a ROS image.
        self.bridge = CvBridge()

        # --- SUBSCRIPTION ---
        # Tell ROS2: "whenever a message arrives on self.input_topic, call self.on_image".
        # In Java this is like: eventBus.subscribe(topicName, Image.class, this::onImage);
        #
        # Arguments:
        #   Image                  = the message type we expect (a camera frame)
        #   self.input_topic       = the topic name, e.g. "/camera/image_raw"
        #   self.on_image          = our callback method (called automatically per frame)
        #   qos_profile_sensor_data = quality-of-service preset: drops stale frames
        self.sub = self.create_subscription(
            Image, self.input_topic, self.on_image, qos_profile_sensor_data
        )

        # --- PUBLISHERS ---
        # Register two output channels we will publish results to.
        #
        # pub_dets: publishes Detection2DArray (the detections with bounding boxes + theta)
        # pub_annot: publishes Image (the camera frame with drawings overlaid)
        #
        # The '10' is the queue size: how many messages to buffer if consumers are slow.
        self.pub_dets = self.create_publisher(
            Detection2DArray, self.output_detections_topic, 10
        )
        self.pub_annot = self.create_publisher(
            Image, self.output_annotated_topic, qos_profile_sensor_data
        )

    # -------------------------------------------------------------------------
    # TASK 1: _load_parameters
    # -------------------------------------------------------------------------
    # This method reads config values from the YAML params file (see config/ folder).
    # The pattern used in all other nodes in this repo:
    #   1. Declare each parameter with its type (STRING, DOUBLE, etc.)
    #   2. Read its value and store it as self.<name>
    #
    # After this method runs, you can use e.g. self.model_path, self.input_topic, etc.
    #
    # The params you need (matching what you'll put in the YAML config file):
    #   'model_path'               - path to the .pt weights file
    #   'confidence_threshold'     - minimum detection confidence (0.0 to 1.0)
    #   'input_topic'              - ROS topic to subscribe to (camera frames)
    #   'output_detections_topic'  - ROS topic to publish Detection2DArray on
    #   'output_annotated_topic'   - ROS topic to publish annotated Image on
    #   'device'                   - inference device: 'cpu', '0', 'cuda', etc.
    #
    # Look at yolo_obb_object_detection_node.py lines 40-51 for the exact pattern.
    def _load_parameters(self):
        # TODO: declare and read parameters here
        # Hint: build a dict of {param_name: Parameter.Type.XXX}, then loop over it.
        # Use self.declare_parameter(name, ptype) then setattr(self, name, self.get_parameter(name).value)
        pass

    # -------------------------------------------------------------------------
    # TASK 2: _load_model
    # -------------------------------------------------------------------------
    # This method loads the YOLO model from the .pt file on disk.
    #
    # Steps:
    #   1. Expand '~' in self.model_path (os.path.expanduser)
    #   2. If the path is not absolute, look for it inside this package's share/model/ dir
    #      (use get_package_share_directory('yolo_ultralytics_pose_node'))
    #   3. Check the file exists — log an error and raise FileNotFoundError if not
    #   4. Log a message saying you are loading the model (self.get_logger().info(...))
    #   5. Return YOLO(mp)  — this loads the weights file
    #
    # Look at yolo_obb_object_detection_node.py lines 53-62 for the exact pattern.
    def _load_model(self):
        # TODO: implement model loading
        pass

    # -------------------------------------------------------------------------
    # TASK 3: on_image  (THE CORE LOGIC — this runs once per camera frame)
    # -------------------------------------------------------------------------
    # This is the callback ROS2 calls every time a new camera frame arrives.
    # 'msg' is a sensor_msgs/Image ROS message.
    #
    # Steps:
    #
    # Step A — Convert the ROS Image message to an OpenCV numpy array:
    #   frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #
    # Step B — Run YOLO inference on the frame:
    #   results = self.model.predict(frame, conf=self.confidence_threshold,
    #                                device=self.device, verbose=False)
    #   This returns a list of result objects (usually just one for a single frame).
    #
    # Step C — Build the output Detection2DArray message:
    #   det_array = Detection2DArray()
    #   det_array.header = msg.header   # carry the original timestamp forward
    #
    # Step D — Loop over detections and extract keypoints:
    #   for r in results:
    #       boxes    = r.boxes           # bounding boxes (one per detected valve)
    #       keypoints = r.keypoints.xy.cpu().numpy()
    #           # shape: [N_detections, N_keypoints, 2]
    #           # e.g. keypoints[i][0] = (x, y) of keypoint 0 for detection i
    #           # e.g. keypoints[i][1] = (x, y) of keypoint 1 for detection i
    #           #   keypoint 0 = handle BASE
    #           #   keypoint 1 = handle TIP
    #
    #   for i, box in enumerate(boxes):
    #       # Read bounding box center and size from box.xywh (format: cx, cy, w, h)
    #       cx, cy, w, h = box.xywh[0].cpu().numpy()
    #       score = float(box.conf[0].cpu())
    #       cid   = int(box.cls[0].cpu())
    #
    #       # Read the two keypoints for this detection
    #       kp0_x, kp0_y = keypoints[i][0]   # handle base
    #       kp1_x, kp1_y = keypoints[i][1]   # handle tip
    #
    #       # Compute yaw angle using atan2 — this is the KEY difference vs OBB!
    #       # atan2(dy, dx) gives the angle of the vector kp0 → kp1
    #       # Range is [-π, π], fully unambiguous (no 180° symmetry problem)
    #       theta = math.atan2(kp1_y - kp0_y, kp1_x - kp0_x)
    #
    # Step E — Build a Detection2D message for this detection and append to det_array:
    #   det = Detection2D()
    #   det.header = msg.header
    #
    #   hyp = ObjectHypothesisWithPose()
    #   hyp.hypothesis.class_id = str(cid)
    #   hyp.hypothesis.score    = score
    #   det.results.append(hyp)
    #
    #   det.bbox = BoundingBox2D()
    #   det.bbox.center.position.x = float(cx)
    #   det.bbox.center.position.y = float(cy)
    #   det.bbox.center.theta      = theta     # <-- computed from keypoints!
    #   det.bbox.size_x = float(w)
    #   det.bbox.size_y = float(h)
    #
    #   det_array.detections.append(det)
    #
    # Step F — Draw annotations on a copy of the frame (for the annotated image topic):
    #   annotated = self.model.predict(...)[0].plot()
    #     OR manually use cv2.circle / cv2.line / cv2.arrowedLine to draw keypoints.
    #   Simplest: use  results[0].plot()  which returns a BGR numpy array with drawings.
    #
    # Step G — Publish both outputs:
    #   self.pub_dets.publish(det_array)
    #
    #   out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
    #   out_msg.header = msg.header
    #   self.pub_annot.publish(out_msg)
    def on_image(self, msg: Image):
        # TODO: implement steps A through G above
        pass


# --- ENTRY POINT ---
# This function is registered as the executable in setup.py.
# In Java this would be:  public static void main(String[] args) { ... }
#
# rclpy.init()       = connect to the ROS2 runtime
# YoloUltralyticsPoseNode() = construct our node (registers subs/pubs)
# rclpy.spin(node)   = block here, running an event loop that calls our callbacks
# destroy + shutdown = cleanup on exit (e.g. Ctrl+C)
def main():
    rclpy.init()
    node = YoloUltralyticsPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
