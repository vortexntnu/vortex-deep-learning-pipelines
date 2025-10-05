#!/usr/bin/env python3
"""ROS 2 node that classifies segmented objects and publishes Detection2DArray."""

import csv
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D

from cv_bridge import CvBridge
import cv2
import numpy as np

from message_filters import Subscriber, ApproximateTimeSynchronizer

class MaskClassifierNode(Node):
    """Node that classifies segmented objects from color+id images."""
    def __init__(self):
        """Initialize subscribers, publishers and load label maps."""
        super().__init__('mask_classifier_node')

        # Parameters
        self.declare_parameter('segmentation_image_color_sub_topic', 'segmentation_image_color')
        self.declare_parameter('segmentation_image_id_sub_topic', 'segmentation_image_id')
        self.declare_parameter('legend_csv_path', '')
        self.declare_parameter('min_pixels_per_object', 200)

        color_topic = self.get_parameter('segmentation_image_color_sub_topic').get_parameter_value().string_value
        id_topic    = self.get_parameter('segmentation_image_id_sub_topic').get_parameter_value().string_value
        self.legend_csv_path = self.get_parameter('legend_csv_path').get_parameter_value().string_value
        self.min_pixels = int(self.get_parameter('min_pixels_per_object').get_parameter_value().integer_value)

        self.bridge = CvBridge()

        # Subscribers (synchronized color + id)
        self.color_sub = Subscriber(self, Image, color_topic, qos_profile=qos_profile_sensor_data)
        self.id_sub    = Subscriber(self, Image, id_topic, qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer([self.color_sub, self.id_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.synchronization_callback)

        # Publishers
        self.scene_pub   = self.create_publisher(String, 'scene/label', 10)
        self.objects_pub = self.create_publisher(Detection2DArray, 'scene/objects', 10)
        self.debug_img_pub = self.create_publisher(Image, 'scene/debug_image', 10)

        # Load legend
        self.id_to_bgr: Dict[int, Tuple[int,int,int]] = {}
        self._load_legend()
        self.get_logger().info('mask_classifier_node ready.')

        # --- Label mapping persistent ---
        home = os.environ.get("HOME", "/tmp")
        default_map = os.path.join(home, "seg_frames", "id_label_map.json")
        self.declare_parameter('id_label_map_path', default_map)
        self.id_label_map_path = self.get_parameter('id_label_map_path').get_parameter_value().string_value

        self.id_to_label = {}                   # dict[int] -> str
        self._label_map_dirty = False           # to know if we need to save
        self._load_or_bootstrap_label_map()     # load or create
        self._prefill_label_map_from_legend()

    def _load_legend(self):
        path = self.legend_csv_path
        if not path or not os.path.exists(path):
            self.get_logger().warn('legend.csv not found (optional). Continuing without id->color map.')
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    _id = int(row['id'])
                    b = int(row['b'])
                    g = int(row['g'])
                    r = int(row['r'])
                    self.id_to_bgr[_id] = (b, g, r)
            self.get_logger().info(f'Loaded legend with {len(self.id_to_bgr)} ids.')
        except (OSError, ValueError, KeyError) as e:
            self.get_logger().warn(f'Could not read legend.csv: {e}')

    def synchronization_callback(self, color_msg: Image, id_msg: Image):
        """Callback with synchronized color and id images."""
        # Convert to OpenCV
        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        ids   = self.bridge.imgmsg_to_cv2(id_msg)  # asume uint16

        if ids.dtype != np.uint16:
            # if comes in another format, try to convert
            ids = ids.astype(np.uint16, copy=False)

        h, w = ids.shape[:2]

        # Count pixels per id
        ids_flat = ids.reshape(-1)
        unique_ids, counts = np.unique(ids_flat, return_counts=True)

        # Filter small objects
        valid = [(int(i), int(c)) for i, c in zip(unique_ids, counts) if c >= self.min_pixels and i != 0]
        if not valid:
            # Publish empty scene
            self._publish_scene('empty')
            self._publish_objects([], color_msg.header)
            return

        # Extract bounding boxes by id (fast with min/max over indices)
        # Create a mask map by id (careful with memory if there are many ids)
        objects: List[Detection2D] = []
        for obj_id, pix in valid:
            mask = ids == obj_id

            ys, xs = np.where(mask)
            if ys.size == 0:
                continue

            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())
            bbox_w = x_max - x_min + 1
            bbox_h = y_max - y_min + 1

            # Hypothesis of class (placeholder): map by mean color or simple rule
            # Here, example: if the mean color is "reddish", we say "valve"; if not, "object".
            label = self._label_for_id(obj_id)

            # Create Detection2D message (BoundingBox2D)
            det = Detection2D()
            det.header = color_msg.header
            det.id = str(obj_id)
            det.bbox = BoundingBox2D()                 # explicit instance
            det.bbox.center.position.x = float(x_min + bbox_w / 2.0)
            det.bbox.center.position.y = float(y_min + bbox_h / 2.0)
            det.bbox.center.theta = 0.0                # if not using rotation
            det.bbox.size_x = float(bbox_w)
            det.bbox.size_y = float(bbox_h)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score = min(1.0, max(0.0, pix / float(h*w)))  # score simple = área relativa
            det.results.append(hyp)

            objects.append(det)

        # Scene classification (placeholder): by total coverage of ids
        coverage = sum(pix for _, pix in valid) / float(h*w)
        scene_label = 'busy' if coverage > 0.5 else 'sparse'
        debug = color.copy()
        for det in objects:
            cx = int(det.bbox.center.position.x)
            cy = int(det.bbox.center.position.y)
            w  = int(det.bbox.size_x)
            h  = int(det.bbox.size_y)
            x1 = max(0, cx - w // 2)
            y1 = max(0, cy - h // 2)
            x2 = min(debug.shape[1]-1, x1 + w)
            y2 = min(debug.shape[0]-1, y1 + h)
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0,255,0), 2)
            # optional: put class text
            if det.results:
                cls = det.results[0].hypothesis.class_id
                cv2.putText(debug, cls, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
        self._publish_scene(scene_label)
        self._publish_objects(objects, color_msg.header)

    def _publish_scene(self, label: str):
        """Publish a high-level scene label as std_msgs/String."""
        msg = String()
        msg.data = label
        self.scene_pub.publish(msg)

    def _publish_objects(self, detections: List[Detection2D], header):
        """Publish a Detection2DArray with computed bounding boxes and scores."""
        arr = Detection2DArray()
        arr.header = header
        arr.detections = detections
        self.objects_pub.publish(arr)

    def _prefill_label_map_from_legend(self):
        """Make sure all IDs seen in legend.csv exist in id_label_map."""
        if not self.id_to_bgr:
            return
        added = 0
        for obj_id in self.id_to_bgr:
            if obj_id not in self.id_to_label:
                if obj_id == 0:
                    label = "background"
                elif obj_id == 65534:
                    label = "unknown"
                else:
                    label = f"id_{obj_id}"
                self.id_to_label[obj_id] = label
                added += 1
        if added:
            self.get_logger().info(f"Prefilled {added} ids into id_label_map from legend.csv.")
            self._save_label_map()

    def _load_or_bootstrap_label_map(self):
        """Load or create id->label map from JSON/CSV file."""
        path = Path(self.id_label_map_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            try:
                if path.suffix.lower() == '.json':
                    with open(path, 'r', encoding='utf-8') as f:
                        raw = json.load(f)
                    # keys can come as str -> convert to int
                    self.id_to_label = {int(k): str(v) for k, v in raw.items()}
                elif path.suffix.lower() == '.csv':
                    with open(path, newline='', encoding='utf-8') as f:
                        for r in csv.DictReader(f):
                            self.id_to_label[int(r['id'])] = str(r['label'])
                else:
                    self.get_logger().warn(f"Extensión no soportada para id_label_map: {path.suffix}, usaré JSON.")
                    with open(path, 'r', encoding='utf-8') as f:
                        raw = json.load(f)
                    self.id_to_label = {int(k): str(v) for k, v in raw.items()}
                self.get_logger().info(f"Cargado id_label_map con {len(self.id_to_label)} entradas de {path}.")
                return
            except (OSError, ValueError, KeyError) as e:
                self.get_logger().warn(f"No pude leer id_label_map {path}: {e}. Creo uno nuevo.")

        # Empty bootstrap (no pre-fill); adding IDs on the fly
        self.id_to_label = {}
        self._save_label_map()  # crea archivo base
        self.get_logger().info(f"Creado id_label_map vacío en {path} (se llenará automáticamente).")

    def _save_label_map(self):
        """Save id->label map to JSON/CSV if dirty."""
        p = Path(self.id_label_map_path)
        try:
            if p.suffix.lower() == '.csv':
                with open(p, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['id', 'label'])
                    for k in sorted(self.id_to_label.keys()):
                        w.writerow([k, self.id_to_label[k]])
            else:  # json by default
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump({str(k): self.id_to_label[k] for k in sorted(self.id_to_label.keys())}, f, indent=2)
            self._label_map_dirty = False
        except (OSError, TypeError, ValueError) as e:
            self.get_logger().warn(f"Error saving id_label_map: {e}")

    def _label_for_id(self, obj_id: int) -> str:
        """Returns the label for an ID. If it doesn't exist, creates it with a default value and persists."""
        if obj_id not in self.id_to_label:
            # default rules
            if obj_id == 0:
                label = "background"
            elif obj_id == 65534:
                label = "unknown"
            else:
                label = f"id_{obj_id}"
            self.id_to_label[obj_id] = label
            self._label_map_dirty = True
            self.get_logger().info(f"New ID seen {obj_id} -> '{label}' added to id_label_map.")
            self._save_label_map()  # save immediately to maintain stability
        return self.id_to_label[obj_id]



def main():
    """Entrypoint: init ROS, spin node, then shutdown."""
    rclpy.init()
    node = MaskClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
