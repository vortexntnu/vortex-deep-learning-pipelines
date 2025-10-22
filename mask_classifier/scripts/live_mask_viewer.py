#!/usr/bin/env python3
import os, json, csv
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def load_label_map(json_path: Path, csv_path: Path):
    """
    Carga el mapa de etiquetas desde:
      1) id_label_map.json (preferido)
         - Formato A: { "labels": [{"id":1,"name":"x"}, ...] }
         - Formato B: { "1": "x", "2": "y", ... }
      2) legend.csv (respaldo): columnas 'id', 'name' (o 'label')
    Devuelve: lista de (id_int, name_str) ordenada por id.
    """
    # 1) JSON primero
    if json_path and json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        label_map = {}
        if isinstance(raw, dict) and "labels" in raw and isinstance(raw["labels"], list):
            for item in raw["labels"]:
                if isinstance(item, dict) and "id" in item and "name" in item:
                    label_map[int(item["id"])] = str(item["name"])
        elif isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    label_map[int(k)] = str(v)
                except Exception:
                    pass
        if label_map:
            return sorted(label_map.items(), key=lambda x: x[0])

    # 2) CSV de respaldo
    if csv_path and csv_path.exists():
        label_map = {}
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # admite 'id', 'name' o 'label'
                rid = row.get("id")
                name = row.get("name") or row.get("label")
                if rid is None or name is None:
                    continue
                try:
                    label_map[int(rid)] = str(name)
                except Exception:
                    pass
        if label_map:
            return sorted(label_map.items(), key=lambda x: x[0])

    raise FileNotFoundError(
        f"No se pudo cargar el mapa de etiquetas. "
        f"Buscado JSON: {json_path} y CSV: {csv_path}"
    )


class LiveMaskViewer(Node):
    def __init__(self):
        super().__init__("live_mask_viewer")

        # --- Rutas por defecto basadas en HOME/seg_frames ---
        home = os.environ.get("HOME", "/tmp")
        seg_frames_dir = Path(home) / "seg_frames"
        default_json = seg_frames_dir / "id_label_map.json"
        default_csv  = seg_frames_dir / "legend.csv"

        # --- Parámetros ROS2 (puedes sobreescribir con --ros-args -p ...) ---
        self.declare_parameter("segmentation_image_id_sub_topic", "/segmentation_image_id")
        self.declare_parameter("id_label_map_path", str(default_json))
        self.declare_parameter("legend_csv_path", str(default_csv))
        self.declare_parameter("window_name", "Live Mask Viewer")

        topic = self.get_parameter("segmentation_image_id_sub_topic").value
        json_path = Path(self.get_parameter("id_label_map_path").value)
        csv_path  = Path(self.get_parameter("legend_csv_path").value)
        self.window_name = self.get_parameter("window_name").value

        # --- Carga labels (JSON preferido; CSV como respaldo) ---
        labels = load_label_map(json_path, csv_path)  # [(id, name)]
        self.labels = labels
        self.selected_index = 0

        # ROS/bridge/estado
        self.bridge = CvBridge()
        self.lock = Lock()
        self.last_mask = None  # np.ndarray mono8 o mono16

        self.sub = self.create_subscription(Image, topic, self.on_mask, 10)
        self.get_logger().info(f"Suscrito a: {topic}")
        self.get_logger().info(f"Label map: {json_path if json_path.exists() else csv_path} · {len(labels)} clases")

        # --- Ventana + “desplegable” sencillo (trackbar) ---
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 720)
        cv2.createTrackbar("Index", self.window_name, 0, max(0, len(self.labels) - 1), self.on_trackbar)

        # Refresco ~30 FPS
        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)
        self.get_logger().info("Controles: ←/→ o A/D para clase · S guardar binaria · Q/Esc salir")

    # --- Callbacks ---
    def on_trackbar(self, val):
        with self.lock:
            self.selected_index = int(np.clip(val, 0, len(self.labels) - 1))

    def on_mask(self, msg: Image):
        try:
            enc = msg.encoding  # respeta mayúsculas
            if enc in ("16UC1", "mono16"):
                # Deja pasar tal cual (16-bit, 1 canal)
                mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.uint16)
            elif enc in ("8UC1", "mono8"):
                mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.uint8)
            else:
                # Otros (por si acaso): intenta convertir a 8 bits
                self.get_logger().warn(f"Encoding {enc} no esperado; intentando mono8.")
                mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8").astype(np.uint8)

            with self.lock:
                self.last_mask = mask
        except Exception as e:
            self.get_logger().warn(f"Error Image->CV ({msg.encoding}): {e}")


    def on_timer(self):
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            rclpy.shutdown(); return
        elif key in (81, ord('a')):  # izq
            with self.lock:
                self.selected_index = (self.selected_index - 1) % len(self.labels)
                cv2.setTrackbarPos("Index", self.window_name, self.selected_index)
        elif key in (83, ord('d')):  # der
            with self.lock:
                self.selected_index = (self.selected_index + 1) % len(self.labels)
                cv2.setTrackbarPos("Index", self.window_name, self.selected_index)
        elif key == ord('s'):
            self.save_current_binary()

        # Render
        with self.lock:
            frame = None if self.last_mask is None else self.last_mask.copy()
            sid, sname = self.labels[self.selected_index]

        if frame is None:
            vis = np.zeros((480, 640, 3), dtype=np.uint8)
            title = "Esperando máscaras..."
        else:
            if frame.dtype == np.uint16:
                binary = np.where(frame == np.uint16(sid), 255, 0).astype(np.uint8)
            else:
                binary = np.where(frame == np.uint8(sid), 255, 0).astype(np.uint8)
            vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            title = f"ID {sid} – {sname}"

        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(vis, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # Listado corto tipo “menú”
        start = max(0, self.selected_index - 4)
        end = min(len(self.labels), self.selected_index + 5)
        y = 60
        for i in range(start, end):
            _id, _name = self.labels[i]
            line = f"{'>' if i==self.selected_index else ' '} [{i}] {_id} – {_name}"
            color = (255,255,255) if i==self.selected_index else (180,180,180)
            cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            y += 18

        cv2.imshow(self.window_name, vis)

    def save_current_binary(self):
        with self.lock:
            if self.last_mask is None: 
                self.get_logger().info("No hay imagen para guardar.")
                return
            sid, _ = self.labels[self.selected_index]
            frame = self.last_mask.copy()

        if frame.dtype == np.uint16:
            binary = np.where(frame == np.uint16(sid), 255, 0).astype(np.uint8)
        else:
            binary = np.where(frame == np.uint8(sid), 255, 0).astype(np.uint8)

        out = Path.cwd() / f"binary_id_{sid}.png"
        cv2.imwrite(str(out), binary)
        self.get_logger().info(f"Guardado: {out}")


def main():
    rclpy.init()
    node = LiveMaskViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
