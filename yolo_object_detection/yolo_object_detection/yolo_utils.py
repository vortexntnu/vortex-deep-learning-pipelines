#!/usr/bin/env python3

import cv2
from ultralytics import YOLO


def load_model(model_path, conf):
    model = YOLO(model_path)
    return model, conf


def process_frame(frame, model, conf, device):
    """Run YOLO inference on an OpenCV frame.

    Returns:
        detections: list of tuples (x1, y1, x2, y2, score, class_id)
        annotated: frame with drawn boxes and labels
    """
    results = model.predict(frame, conf=conf, verbose=False, device=device)
    detections = []
    annotated = frame.copy()

    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue

        b = r.boxes
        xyxy = b.xyxy.cpu().numpy() if hasattr(b.xyxy, "cpu") else b.xyxy
        confs = b.conf.cpu().numpy() if hasattr(b.conf, "cpu") else b.conf
        clss = b.cls.cpu().numpy() if hasattr(b.cls, "cpu") else b.cls

        for (x1, y1, x2, y2), sc, cid in zip(xyxy, confs, clss):
            detections.append((x1, y1, x2, y2, float(sc), int(cid)))

            # Draw annotation
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
            label = f"{model.names.get(int(cid), str(int(cid)))} {sc:.2f}"
            cv2.putText(
                annotated,
                label,
                (p1[0], max(p1[1] - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    return detections, annotated
