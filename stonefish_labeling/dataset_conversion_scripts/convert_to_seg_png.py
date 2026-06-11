#!/usr/bin/env python3
import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_id_to_label(seg_dir: Path) -> dict:
    j = seg_dir / "id_label_map.json"
    if j.exists():
        raw = json.loads(j.read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in raw.items()}
    # basic fallback
    legend = seg_dir / "legend.csv"
    labels = {}
    if legend.exists():
        for i in pd.read_csv(legend)["id"].tolist():
            labels[int(i)] = (
                "background" if i == 0 else ("unknown" if i == 65534 else f"id_{i}")
            )
    return labels


def load_legend_colors(seg_dir: Path) -> dict:
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
    if ids_img is None:
        return None
    if ids_img.ndim == 2:
        return ids_img.astype(np.int32)
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


def discover_classes(seg_dir: Path, min_pixels: int, legend_colors: dict) -> list[int]:
    present = Counter()
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
            if obj_id in (0, 65534) or int(cnt) < min_pixels:
                continue
            present[obj_id] += int(cnt)
    return sorted(present.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-pixels", type=int, default=200)
    ap.add_argument(
        "--pipeline-ids",
        type=str,
        default=None,
        help="Comma-separated list of object IDs to merge into a single 'pipeline' class (e.g. 1,4,5,6,7,8,9). All other IDs become background.",
    )
    args = ap.parse_args()

    seg_dir = Path(os.path.expanduser(args.seg_dir))
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    legend_colors = load_legend_colors(seg_dir)

    if args.pipeline_ids is not None:
        pipeline_id_set = {int(x.strip()) for x in args.pipeline_ids.split(",")}
        (out_dir / "classes.txt").write_text("pipeline", encoding="utf-8")
        id2idx = None  # signal: use pipeline mode
    else:
        id2label = load_id_to_label(seg_dir)
        class_ids = discover_classes(seg_dir, args.min_pixels, legend_colors)
        classes = [id2label.get(i, f"id_{i}") for i in class_ids]
        (out_dir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
        id2idx = {cid: i for i, cid in enumerate(class_ids)}
        pipeline_id_set = None

    # Pair front images and masks (frame_*.png and frame_*_mask.*)
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

        idm = convert_mask_image_to_ids(ids_raw, legend_colors)
        if idm is None:
            print(f"WARNING: cannot convert mask for {cpath.name}")
            continue

        h, w = idm.shape[:2]
        mask_out = np.zeros((h, w), dtype=np.uint8)
        if pipeline_id_set is not None:
            for obj_id in pipeline_id_set:
                mask_out[idm == obj_id] = 1  # 1 = pipeline, 0 = background
        else:
            for obj_id, cls_idx in id2idx.items():
                mask_out[idm == obj_id] = cls_idx + 1  # reserve 0 for background

        # save
        cv2.imwrite(str(out_dir / "images" / cpath.name), img)
        cv2.imwrite(str(out_dir / "masks" / (stem + ".png")), mask_out)

    print(f"Done. Exported to: {out_dir}")
    print(
        "Upload 'images/' and 'masks/' to Roboflow as 'Semantic Segmentation (PNG masks)'."
    )


if __name__ == "__main__":
    main()
