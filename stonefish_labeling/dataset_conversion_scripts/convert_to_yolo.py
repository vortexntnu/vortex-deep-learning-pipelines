#!/usr/bin/env python3
import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_id_to_label(seg_dir: Path) -> dict:
    # 1) explicit JSON, 2) CSV (id,label), 3) fallback: default rules
    json_path = seg_dir / "id_label_map.json"
    csv_path = seg_dir / "id_label_map.csv"
    if json_path.exists():
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in raw.items()}
    if csv_path.exists():
        m = {}
        for r in pd.read_csv(csv_path).to_dict("records"):
            m[int(r["id"])] = str(r["label"])
        return m
    # Fallback: infer from legend.csv or dataset when no explicit map
    m = {}
    legend = seg_dir / "legend.csv"
    if legend.exists():
        ids = pd.read_csv(legend)["id"].tolist()
        for i in ids:
            if i == 0:
                m[i] = "background"
            elif i == 65534:
                m[i] = "unknown"
            else:
                m[i] = f"id_{i}"
    return m


def load_legend_colors(seg_dir: Path) -> dict:
    """Load legend.csv and return mapping id -> (r,g,b)."""
    legend_path = seg_dir / "legend.csv"
    colors = {}
    if not legend_path.exists():
        return colors
    df = pd.read_csv(legend_path)
    for _, row in df.iterrows():
        try:
            i = int(row["id"])
            r = int(row["r"])
            g = int(row["g"])
            b = int(row["b"])
            colors[i] = (r, g, b)
        except Exception as e:
            logging.debug("Skipping legend row due to error: %s", e)
            continue
    return colors


def convert_mask_image_to_ids(ids_img: np.ndarray, legend_colors: dict) -> np.ndarray:
    """Convert a mask image (single-channel or color) to integer id map.

    Unknown colors map to 65534.
    """
    if ids_img is None:
        return None
    if ids_img.ndim == 2:
        return ids_img.astype(np.int32)

    # Build color->id map (BGR codes since OpenCV reads BGR)
    color_to_id = {}
    for _id, (r, g, b) in legend_colors.items():
        code = (b & 0xFF) | ((g & 0xFF) << 8) | ((r & 0xFF) << 16)
        color_to_id[code] = _id

    flat = ids_img.reshape(-1, ids_img.shape[2])[:, :3]
    codes = (
        flat[:, 0].astype(np.uint32)
        | (flat[:, 1].astype(np.uint32) << 8)
        | (flat[:, 2].astype(np.uint32) << 16)
    )
    mapped = np.full(codes.shape, 65534, dtype=np.int32)
    for code, _id in color_to_id.items():
        mapped[codes == code] = _id
    return mapped.reshape(ids_img.shape[0], ids_img.shape[1])


def yolo_line(cx, cy, w, h, img_w, img_h, cls_idx):
    return (
        f"{cls_idx} {cx / img_w:.6f} {cy / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seg-dir",
        required=True,
        help="Directory with masks and legend (e.g. frame_*.png, frame_*_mask.tiff)",
    )
    ap.add_argument(
        "--out-dir", required=True, help="Output directory for YOLO dataset"
    )
    ap.add_argument(
        "--min-pixels", type=int, default=200, help="Minimum pixels per object to keep"
    )
    args = ap.parse_args()

    seg_dir = Path(os.path.expanduser(args.seg_dir))
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Load id->label mapping
    id2label = load_id_to_label(seg_dir)

    # Load legend colors for color mask mapping
    legend_colors = load_legend_colors(seg_dir)

    # Discover present IDs by scanning mask files (supports *_mask.* and legacy *_ids.tiff)
    present_ids = set()
    mask_files = sorted(seg_dir.glob("*_mask.*"))
    if not mask_files:
        mask_files = sorted(seg_dir.glob("*_ids.tiff"))

    for mf in mask_files:
        ids_raw = cv2.imread(str(mf), cv2.IMREAD_UNCHANGED)
        if ids_raw is None:
            continue
        ids_map = convert_mask_image_to_ids(ids_raw, legend_colors)
        if ids_map is None:
            continue
        uniq, counts = np.unique(ids_map, return_counts=True)
        for obj_id, cnt in zip(uniq, counts):
            obj_id = int(obj_id)
            if obj_id in (0, 65534):
                continue
            if int(cnt) >= args.min_pixels:
                present_ids.add(obj_id)

    # Stable order
    class_ids = sorted(present_ids)

    # Labels (fallback to id_{id} when name missing)
    classes = [id2label.get(obj_id, f"id_{obj_id}") for obj_id in class_ids]

    # Save classes file
    (out_dir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")

    # id -> YOLO index map (0..N-1)
    id2idx = {cid: i for i, cid in enumerate(class_ids)}

    # Convert each frame: pair front images (frame_*.png) with masks (frame_*_mask.tiff/png)
    front_files = sorted(seg_dir.glob("frame_*.png"))
    if not front_files:
        front_files = sorted(seg_dir.glob("*.png"))

    for cpath in front_files:
        img = cv2.imread(str(cpath), cv2.IMREAD_COLOR)
        stem = cpath.stem
        mask_tiff = seg_dir / f"{stem}_mask.tiff"
        mask_png = seg_dir / f"{stem}_mask.png"
        if mask_tiff.exists():
            ids_raw = cv2.imread(str(mask_tiff), cv2.IMREAD_UNCHANGED)
        elif mask_png.exists():
            ids_raw = cv2.imread(str(mask_png), cv2.IMREAD_UNCHANGED)
        else:
            legacy = seg_dir / f"{stem}_ids.tiff"
            ids_raw = (
                cv2.imread(str(legacy), cv2.IMREAD_UNCHANGED)
                if legacy.exists()
                else None
            )

        if ids_raw is None or img is None:
            print(f"WARNING: Skipping {cpath.name} (failed to read front or mask)")
            continue

        ids = convert_mask_image_to_ids(ids_raw, legend_colors)
        if ids is None:
            print(f"WARNING: Skipping {cpath.name} (cannot convert mask)")
            continue

        h, w = ids.shape[:2]
        flat = ids.reshape(-1)
        uniq, counts = np.unique(flat, return_counts=True)

        lines = []
        for obj_id, count in zip(uniq, counts):
            obj_id = int(obj_id)
            if obj_id in (0, 65534):
                continue
            if count < args.min_pixels:
                continue
            if obj_id not in id2idx:
                # id present but filtered globally; ignore
                continue

            mask = ids == obj_id
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            w_box = x_max - x_min + 1
            h_box = y_max - y_min + 1
            cx = x_min + w_box / 2
            cy = y_min + h_box / 2

            cls_idx = id2idx[obj_id]
            lines.append(yolo_line(cx, cy, w_box, h_box, w, h, cls_idx))

        # write label
        label_path = out_dir / "labels" / (stem + ".txt")
        label_path.write_text("\n".join(lines), encoding="utf-8")

        # copy corresponding image into output images folder
        try:
            shutil.copy2(cpath, out_dir / "images" / cpath.name)
        except Exception as e:
            logging.debug("Failed to copy image %s: %s", cpath, e)

    # minimal data.yaml for YOLO/Roboflow
    yaml = [
        f"path: {out_dir.resolve()}",
        "train: images",
        "val: images",  # if not split, Roboflow will re-split on import
        f"names: {classes}",
    ]
    (out_dir / "data.yaml").write_text("\n".join(yaml), encoding="utf-8")
    print(f"Done. Exported to: {out_dir}")


if __name__ == "__main__":
    main()
