#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO


def load_model(model_path, conf):
    return YOLO(model_path)


def _draw_obb(img, cx, cy, w, h, theta, color=(0, 255, 0), thickness=2):
    rect = (
        (float(cx), float(cy)),
        (float(w), float(h)),
        float(np.degrees(theta)),
    )
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.polylines(img, [box], True, color, thickness)
    return box


def process_frame(frame, model, conf, device):
    results = model.predict(frame, conf=conf, device=device, verbose=False)

    detections = []
    annotated = frame.copy()

    for r in results:
        if r.obb is None:
            raise RuntimeError(
                "Loaded model does not output OBB predictions. "
                "Make sure you are using a YOLO26-OBB model."
            )

        xywhr = r.obb.xywhr.cpu().numpy()
        confs = r.obb.conf.cpu().numpy()
        clss = r.obb.cls.cpu().numpy()

        for (cx, cy, w, h, theta), score, cls_id in zip(xywhr, confs, clss):
            cls_id = int(cls_id)
            score = float(score)
            theta = float(theta)

            detections.append(
                (float(cx), float(cy), float(w), float(h), theta, score, cls_id)
            )

            box = _draw_obb(annotated, cx, cy, w, h, theta)

            label = f"{model.names[cls_id]} {score:.2f}"
            x0, y0 = int(box[0][0]), int(box[0][1])

            cv2.putText(
                annotated,
                label,
                (x0, max(y0 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    return detections, annotated
