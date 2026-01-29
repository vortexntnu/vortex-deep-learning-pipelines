#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO


def load_model(model_path, conf):
    return YOLO(model_path)


def _draw_obb(annotated, cx, cy, w, h, theta, color=(0, 255, 0), thickness=2):
    rect = ((float(cx), float(cy)), (float(w), float(h)), float(np.degrees(theta)))
    box = cv2.boxPoints(rect)  # 4x2
    box = np.intp(box)
    cv2.polylines(annotated, [box], isClosed=True, color=color, thickness=thickness)
    return box


def process_frame(frame, model, conf, device):
    results = model.predict(frame, conf=conf, verbose=False, device=device)
    detections = []
    annotated = frame.copy()

    for r in results:
        # ---- Prefer OBB outputs if present ----
        if hasattr(r, "obb") and r.obb is not None:
            obb = r.obb
            xywhr = (
                obb.xywhr.cpu().numpy()
                if hasattr(obb.xywhr, "cpu")
                else np.asarray(obb.xywhr)
            )
            confs = (
                obb.conf.cpu().numpy()
                if hasattr(obb.conf, "cpu")
                else np.asarray(obb.conf)
            )
            clss = (
                obb.cls.cpu().numpy()
                if hasattr(obb.cls, "cpu")
                else np.asarray(obb.cls)
            )

            for (cx, cy, w, h, theta), sc, cid in zip(xywhr, confs, clss):
                cid_i = int(cid)
                sc_f = float(sc)
                theta_f = float(theta)

                detections.append(
                    (float(cx), float(cy), float(w), float(h), theta_f, sc_f, cid_i)
                )

                # Draw rotated box + label
                box = _draw_obb(annotated, cx, cy, w, h, theta_f)
                label = f"{model.names.get(cid_i, str(cid_i))} {sc_f:.2f}"

                # Put label near first corner
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
            continue

        # ---- Fallback to axis-aligned boxes ----
        if not hasattr(r, "boxes") or r.boxes is None:
            continue

        b = r.boxes
        xyxy = b.xyxy.cpu().numpy() if hasattr(b.xyxy, "cpu") else np.asarray(b.xyxy)
        confs = b.conf.cpu().numpy() if hasattr(b.conf, "cpu") else np.asarray(b.conf)
        clss = b.cls.cpu().numpy() if hasattr(b.cls, "cpu") else np.asarray(b.cls)

        for (x1, y1, x2, y2), sc, cid in zip(xyxy, confs, clss):
            cid_i = int(cid)
            sc_f = float(sc)

            detections.append((float(x1), float(y1), float(x2), float(y2), sc_f, cid_i))

            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
            label = f"{model.names.get(cid_i, str(cid_i))} {sc_f:.2f}"
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
