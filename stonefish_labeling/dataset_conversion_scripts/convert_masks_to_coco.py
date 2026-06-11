#!/usr/bin/env python3
"""Convert a PNG-mask segmentation dataset to COCO instance segmentation JSON and zip it ready for Roboflow upload.

Each connected blob in the mask becomes one instance annotation.

Usage:
    python3 convert_masks_to_coco.py \
        --dataset-dir /home/kluge7/stonefish_seg_pipeline \
        --out-zip     /home/kluge7/stonefish_coco.zip \
        --min-pixels  200
"""

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

CATEGORIES = [{"id": 1, "name": "pipeline", "supercategory": ""}]


def mask_to_instances(mask: np.ndarray, min_pixels: int):
    """Return list of (polygon_flat, bbox, area) for each connected blob in mask."""
    binary = (mask == 1).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    instances = []
    for label_id in range(1, num_labels):  # 0 is background
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue
        instance_mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            continue
        polygon = contour.flatten().tolist()
        if len(polygon) < 6:  # COCO needs at least 3 points
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        instances.append((polygon, [x, y, w, h], area))
    return instances


def build_coco(dataset_dir: Path, min_pixels: int):
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"

    coco = {
        "info": {"description": "Pipeline segmentation dataset"},
        "licenses": [],
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }

    annotation_id = 1
    image_files = sorted(images_dir.glob("*.png"))

    for image_id, img_path in enumerate(image_files, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARNING: could not read {img_path.name}, skipping")
            continue
        h, w = img.shape[:2]
        coco["images"].append(
            {"id": image_id, "file_name": img_path.name, "width": w, "height": h}
        )

        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            print(f"WARNING: no mask for {img_path.name}, skipping annotations")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue

        for polygon, bbox, area in mask_to_instances(mask, min_pixels):
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [polygon],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return coco, image_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-dir", required=True, help="Folder with images/ and masks/"
    )
    ap.add_argument("--out-zip", required=True, help="Output zip path")
    ap.add_argument(
        "--min-pixels", type=int, default=200, help="Minimum blob size to keep"
    )
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_zip = Path(args.out_zip)

    print("Building COCO annotations...")
    coco, image_files = build_coco(dataset_dir, args.min_pixels)

    n_images = len(coco["images"])
    n_annotations = len(coco["annotations"])
    print(f"  {n_images} images, {n_annotations} instance annotations")

    print(f"Writing zip to {out_zip} ...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("_annotations.coco.json", json.dumps(coco, indent=2))
        for img_path in image_files:
            zf.write(img_path, img_path.name)

    print(
        f"Done. Upload '{out_zip}' to Roboflow as an Instance Segmentation project (COCO format)."
    )


if __name__ == "__main__":
    main()
