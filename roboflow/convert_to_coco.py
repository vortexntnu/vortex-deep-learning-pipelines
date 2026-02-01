#!/usr/bin/env python3
"""
Convert segmentation data to COCO JSON format for Roboflow upload.
Extracts polygon contours from mask images for instance segmentation.
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import cv2
import pandas as pd


def load_id_to_label(seg_dir: Path) -> dict:
    """Load ID to label mapping from id_label_map.json or fallback to legend.csv"""
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


def discover_classes(seg_dir: Path, min_pixels: int) -> list[int]:
    """Discover which class IDs are present across all frames"""
    present = Counter()
    for stats_file in sorted(seg_dir.glob("*_stats.csv")):
        df = pd.read_csv(stats_file)
        for _, row in df.iterrows():
            obj_id = int(row["id"])
            pixels = int(row["pixels"])
            # Skip background and unknown
            if obj_id in (0, 65534) or pixels < min_pixels:
                continue
            present[obj_id] += pixels
    return sorted(present.keys())


def extract_polygon_from_mask(mask: np.ndarray, tolerance: float = 2.0) -> list:
    """
    Extract polygon coordinates from a binary mask.
    Returns list of [x1,y1,x2,y2,...] for COCO format.
    """
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    """Compute bounding box [x, y, width, height] from polygon"""
    xs = polygon[0::2]
    ys = polygon[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def compute_area_from_polygon(polygon: list) -> float:
    """Compute polygon area using shoelace formula"""
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
    parser = argparse.ArgumentParser(description="Convert segmentation masks to COCO JSON format")
    parser.add_argument("--seg-dir", required=True, help="Directory with *_color.png, *_ids.tiff, legend.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory for COCO dataset")
    parser.add_argument("--min-pixels", type=int, default=200, help="Minimum pixels per object instance")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Polygon simplification tolerance")
    args = parser.parse_args()

    seg_dir = Path(os.path.expanduser(args.seg_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    # Load ID to label mapping
    id2label = load_id_to_label(seg_dir)

    # Discover present classes
    class_ids = discover_classes(seg_dir, args.min_pixels)
    if not class_ids:
        print("ERROR: No valid classes found in dataset!")
        return

    # Create category mapping
    categories = []
    id2category = {}
    for idx, obj_id in enumerate(class_ids, start=1):
        label = id2label.get(obj_id, f"id_{obj_id}")
        categories.append({
            "id": idx,
            "name": label,
            "supercategory": "object"
        })
        id2category[obj_id] = idx

    print(f"Found {len(categories)} classes: {[c['name'] for c in categories]}")

    # Save classes.txt for Roboflow labelmap
    classes_txt = out_dir / "classes.txt"
    class_names = [c['name'] for c in categories]
    classes_txt.write_text("\n".join(class_names), encoding="utf-8")
    print(f"Created labelmap: {classes_txt}")

    # Find all frame pairs (use front camera images for training, not segmentation color)
    front_files = sorted(seg_dir.glob("*_front.png"))
    ids_files = sorted(seg_dir.glob("*_ids.tiff"))

    if len(front_files) != len(ids_files):
        print(f"WARNING: Mismatch in file counts: {len(front_files)} front vs {len(ids_files)} ids")

    assert len(front_files) > 0, "No frames found!"

    # COCO structure
    coco_data = {
        "info": {
            "description": "Underwater segmentation dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Vortex Deep Learning Pipeline",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories
    }

    annotation_id = 1

    # Process each frame
    for img_id, (front_path, ids_path) in enumerate(zip(front_files, ids_files), start=1):
        # Read images
        front_img = cv2.imread(str(front_path))
        ids_img = cv2.imread(str(ids_path), cv2.IMREAD_UNCHANGED)  # uint16

        if front_img is None or ids_img is None:
            print(f"WARNING: Skipping {front_path.name} (failed to read)")
            continue

        height, width = front_img.shape[:2]

        # Copy front camera image to output (actual RGB image for training)
        out_img_name = f"frame_{img_id:06d}.png"
        cv2.imwrite(str(out_dir / "images" / out_img_name), front_img)

        # Add image info
        coco_data["images"].append({
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": out_img_name,
            "license": 0,
            "date_captured": ""
        })

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
                print(f"WARNING: Could not extract polygon for ID {obj_id} in {front_path.name}")
                continue

            # Compute bbox and area
            bbox = compute_bbox_from_polygon(polygon)
            area = compute_area_from_polygon(polygon)

            # Add annotation
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": id2category[obj_id],
                "segmentation": [polygon],  # COCO expects list of polygons
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })
            annotation_id += 1

        if img_id % 10 == 0:
            print(f"Processed {img_id}/{len(front_files)} frames...")

    # Save COCO JSON
    coco_path = out_dir / "annotations.json"
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)

    print(f"\n✓ Conversion complete!")
    print(f"  Output directory: {out_dir}")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(categories)}")
    print(f"\nFiles created:")
    print(f"  - {coco_path}")
    print(f"  - {classes_txt}")
    print(f"  - {out_dir / 'images'} (folder with {len(coco_data['images'])} images)")
    print(f"\nTo upload to Roboflow:")
    print(f"  1. Create a new 'Instance Segmentation' project")
    print(f"  2. Upload using 'COCO JSON' format")
    print(f"  3. Upload: {coco_path}")
    print(f"  4. Upload images folder: {out_dir / 'images'}")
    print(f"  5. If asked for labelmap, upload: {classes_txt}")


if __name__ == "__main__":
    main()
