#!/usr/bin/env python3

import cv2
from PIL import Image as PILImage
from ultralytics import YOLO


def load_model(model_path):
    return YOLO(model_path)


def process_frame(frame, model, imgsz, device, verbose):
    """Run YOLO classification on an OpenCV BGR frame.

    Returns:
        class_id: int, top-1 predicted class index
        conf: float, top-1 confidence
        class_name: str, human-readable class name
    """
    pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = model(pil_image, imgsz=imgsz, device=device, verbose=verbose)

    probs = results[0].probs
    class_id = int(probs.top1)
    conf = float(probs.top1conf)
    class_name = model.names[class_id]

    return class_id, conf, class_name
