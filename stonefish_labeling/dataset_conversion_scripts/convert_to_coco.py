#!/usr/bin/env python3
"""Convert segmentation data to COCO JSON format for Roboflow upload.

Extracts polygon contours from mask images for instance segmentation.
"""

import argparse
import json
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_id_to_label(seg_dir: Path) -> dict:
    """Load ID to label mapping from id_label_map.json or fallback to legend.csv."""
    json_path = seg_dir / "id_label_map.json"
    if json_path.exists():
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in raw.items()}

    # Fallback to legend.csv
    legend_path = seg_dir / "legend.csv"
    labels = {}
    if legend_path.exists():
        df = pd.read_csv(legend_path)
        for obj_id in df["id"].unique():
            i = int(obj_id)
            if i == 0:
                labels[i] = "background"
            elif i == 65534:
                labels[i] = "unknown"
            else:
                labels[i] = f"id_{i}"
    return labels


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


def discover_classes(seg_dir: Path, min_pixels: int, legend_colors: dict) -> list[int]:
    """Discover which class IDs are present across all mask images in the refactored dataset.

    The dataset now stores masks as <stem>_mask.tiff where the mask can be either a
    single-channel id map (uint8/uint16) or a color image where each object id maps
    to a color in `legend.csv`.
    """
    present = Counter()

    # Find mask files
    mask_files = sorted(seg_dir.glob("*_mask.*"))
    if not mask_files:
        # Fallback to *_ids.tiff legacy pattern
        mask_files = sorted(seg_dir.glob("*_ids.tiff"))

    for mask_path in mask_files:
        # Try reading
        ids_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if ids_img is None:
            continue

        # Convert to id map (uint32) using legend colors if needed
        if ids_img.ndim == 2:
            unique_ids, counts = np.unique(ids_img, return_counts=True)
        else:
            # color image (BGR from OpenCV) -> map to ids via legend_colors
            # build code mapping for speed
            # legend colors are stored as (r,g,b)
            color_to_id = {}
            for _id, (r, g, b) in legend_colors.items():
                code = (b & 0xFF) | ((g & 0xFF) << 8) | ((r & 0xFF) << 16)
                color_to_id[code] = _id

            h, w = ids_img.shape[:2]
            flat = ids_img.reshape(-1, ids_img.shape[2])[:, :3]
            codes = (
                flat[:, 0].astype(np.uint32)
                | (flat[:, 1].astype(np.uint32) << 8)
                | (flat[:, 2].astype(np.uint32) << 16)
            )
            # Map codes to ids; unknown colors -> 65534
            mapped = np.full(codes.shape, 65534, dtype=np.int32)
            for code, _id in color_to_id.items():
                mapped[codes == code] = _id
            unique_ids, counts = np.unique(mapped, return_counts=True)

        for obj_id, pixels in zip(unique_ids, counts):
            obj_id = int(obj_id)
            if obj_id in (0, 65534) or pixels < min_pixels:
                continue
            present[obj_id] += int(pixels)

    return sorted(present.keys())


def convert_mask_image_to_ids(ids_img: np.ndarray, legend_colors: dict) -> np.ndarray:
    """Convert a mask image to an integer id map.

    - If ids_img is single-channel, return it as-is (cast to int32).
    - If ids_img is color (BGR), map pixels to ids using legend_colors (r,g,b).
      Unknown colors will be mapped to 65534.
    """
    if ids_img is None:
        return None
    if ids_img.ndim == 2:
        return ids_img.astype(np.int32)

    # Build color->id map (codes based on BGR as read by OpenCV)
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


def extract_polygon_from_mask(mask: np.ndarray, tolerance: float = 2.0) -> list:
    """Extract polygon coordinates from a binary mask.

    Returns list of [x1,y1,x2,y2,...] for COCO format.
    """
    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return []

    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Simplify polygon to reduce point count
    epsilon = tolerance
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Flatten to COCO format: [x1, y1, x2, y2, ...]
    polygon = approx.flatten().tolist()

    # COCO requires at least 6 coordinates (3 points)
    if len(polygon) < 6:
        return []

    return polygon


def compute_bbox_from_polygon(polygon: list) -> list:
    """Compute bounding box [x, y, width, height] from polygon."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def compute_area_from_polygon(polygon: list) -> float:
    """Compute polygon area using shoelace formula."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0


def main():
    parser = argparse.ArgumentParser(
        description="Convert segmentation masks to COCO JSON format"
    )
    parser.add_argument(
        "--seg-dir",
        required=True,
        help="Directory with *_color.png, *_ids.tiff, legend.csv",
    )
    parser.add_argument(
        "--out-dir", required=True, help="Output directory for COCO dataset"
    )
    parser.add_argument(
        "--min-pixels", type=int, default=200, help="Minimum pixels per object instance"
    )
    parser.add_argument(
        "--tolerance", type=float, default=2.0, help="Polygon simplification tolerance"
    )
    args = parser.parse_args()

    seg_dir = Path(os.path.expanduser(args.seg_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    # Load ID to label mapping
    id2label = load_id_to_label(seg_dir)

    # Load legend colors (for color masks -> id mapping)
    legend_colors = load_legend_colors(seg_dir)

    # Discover present classes
    class_ids = discover_classes(seg_dir, args.min_pixels, legend_colors)
    if not class_ids:
        print("ERROR: No valid classes found in dataset!")
        return

    # Create category mapping
    categories = []
    id2category = {}
    for idx, obj_id in enumerate(class_ids, start=1):
        label = id2label.get(obj_id, f"id_{obj_id}")
        categories.append({"id": idx, "name": label, "supercategory": "object"})
        id2category[obj_id] = idx

    print(f"Found {len(categories)} classes: {[c['name'] for c in categories]}")

    # Save classes.txt for Roboflow labelmap
    classes_txt = out_dir / "classes.txt"
    class_names = [c['name'] for c in categories]
    classes_txt.write_text("\n".join(class_names), encoding="utf-8")
    print(f"Created labelmap: {classes_txt}")

    # Find all frame pairs (front image + corresponding *_mask.*)
    front_files = sorted(
        [p for p in seg_dir.glob("frame_*.png") if not p.name.endswith("_mask.png")]
    )
    if not front_files:
        # fallback to any png that doesn't look like legend
        front_files = sorted(seg_dir.glob("*.png"))

    assert len(front_files) > 0, "No frames found!"

    # COCO structure
    coco_data = {
        "info": {
            "description": "Underwater segmentation dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Vortex Deep Learning Pipeline",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    annotation_id = 1

    # Process each frame
    for img_id, front_path in enumerate(front_files, start=1):
        # Read front image
        front_img = cv2.imread(str(front_path))
        # Corresponding mask path: same stem + _mask.* (prefer tiff)
        stem = front_path.stem
        mask_tiff = seg_dir / f"{stem}_mask.tiff"
        mask_png = seg_dir / f"{stem}_mask.png"
        if mask_tiff.exists():
            mask_path = mask_tiff
        elif mask_png.exists():
            mask_path = mask_png
        else:
            print(f"WARNING: No mask for {front_path.name}, skipping")
            continue

        ids_img_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if front_img is None or ids_img_raw is None:
            print(f"WARNING: Skipping {front_path.name} (failed to read front or mask)")
            continue

        # Convert mask (color or single-channel) to id map
        ids_img = convert_mask_image_to_ids(ids_img_raw, legend_colors)

        height, width = front_img.shape[:2]

        # Copy front camera image to output (actual RGB image for training)
        out_img_name = f"frame_{img_id:06d}.png"
        cv2.imwrite(str(out_dir / "images" / out_img_name), front_img)

        # Add image info
        coco_data["images"].append(
            {
                "id": img_id,
                "width": width,
                "height": height,
                "file_name": out_img_name,
                "license": 0,
                "date_captured": "",
            }
        )

        # Process each object instance in this frame
        unique_ids, counts = np.unique(ids_img, return_counts=True)

        for obj_id, pixel_count in zip(unique_ids, counts):
            obj_id = int(obj_id)

            # Skip background, unknown, and small objects
            if obj_id in (0, 65534) or pixel_count < args.min_pixels:
                continue

            # Skip if not in our class list
            if obj_id not in id2category:
                continue

            # Create binary mask for this instance
            mask = (ids_img == obj_id).astype(np.uint8)

            # Extract polygon
            polygon = extract_polygon_from_mask(mask, args.tolerance)

            if not polygon:
                print(
                    f"WARNING: Could not extract polygon for ID {obj_id} in {front_path.name}"
                )
                continue

            # Compute bbox and area
            bbox = compute_bbox_from_polygon(polygon)
            area = compute_area_from_polygon(polygon)

            # Add annotation
            coco_data["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": id2category[obj_id],
                    "segmentation": [polygon],  # COCO expects list of polygons
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        if img_id % 10 == 0:
            print(f"Processed {img_id}/{len(front_files)} frames...")

    # Save COCO JSON
    coco_path = out_dir / "annotations.json"
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)

    print("\n✓ Conversion complete!")
    print(f"  Output directory: {out_dir}")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(categories)}")
    print("\nFiles created:")
    print(f"  - {coco_path}")
    print(f"  - {classes_txt}")
    print(f"  - {out_dir / 'images'} (folder with {len(coco_data['images'])} images)")
    print("\nTo upload to Roboflow:")
    print("  1. Create a new 'Instance Segmentation' project")
    print("  2. Upload using 'COCO JSON' format")
    print(f"  3. Upload: {coco_path}")
    print(f"  4. Upload images folder: {out_dir / 'images'}")
    print(f"  5. If asked for labelmap, upload: {classes_txt}")


if __name__ == "__main__":
    main()
