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


def obb_line_from_mask(
    mask: np.ndarray, img_w: int, img_h: int, cls_idx: int, axis_aligned: bool = False
):
    """Return YOLO-OBB line: 'cls x1 y1 x2 y2 x3 y3 x4 y4' (normalized)."""
    ys, xs = np.where(mask)
    if ys.size < 3:
        return None
    if axis_aligned:
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        box = np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            dtype=np.float32,
        )
    else:
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
    coords = []
    for x, y in box:
        coords.append(f"{x / img_w:.6f}")
        coords.append(f"{y / img_h:.6f}")
    return f"{cls_idx} " + " ".join(coords)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-pixels", type=int, default=200)
    ap.add_argument(
        "--keep-ids",
        type=str,
        default="",
        help="Comma-separated list of ids to keep (e.g. '7'). Empty = keep all.",
    )
    ap.add_argument(
        "--axis-aligned-ids",
        type=str,
        default="",
        help="Comma-separated ids that should get axis-aligned (upright) boxes "
        "instead of minAreaRect. Still emitted in OBB format.",
    )
    args = ap.parse_args()

    keep_ids = (
        {int(x) for x in args.keep_ids.split(",") if x.strip()}
        if args.keep_ids
        else None
    )
    axis_aligned_ids = {int(x) for x in args.axis_aligned_ids.split(",") if x.strip()}

    seg_dir = Path(os.path.expanduser(args.seg_dir))
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    id2label = load_id_to_label(seg_dir)
    legend_colors = load_legend_colors(seg_dir)

    present_ids = set()
    mask_files = sorted(seg_dir.glob("*_mask.*"))
    if not mask_files:
        mask_files = sorted(seg_dir.glob("*_ids.tiff"))

    total_masks = len(mask_files)
    print(f"[1/2] Scanning {total_masks} masks to discover class ids...")
    for i, mf in enumerate(mask_files, 1):
        if i % 50 == 0 or i == total_masks:
            print(f"  scanned {i}/{total_masks}")
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
            if keep_ids is not None and obj_id not in keep_ids:
                continue
            if int(cnt) >= args.min_pixels:
                present_ids.add(obj_id)

    class_ids = sorted(present_ids)
    classes = [id2label.get(obj_id, f"id_{obj_id}") for obj_id in class_ids]
    (out_dir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    id2idx = {cid: i for i, cid in enumerate(class_ids)}

    front_files = sorted(seg_dir.glob("frame_*.png"))
    if not front_files:
        front_files = sorted(seg_dir.glob("*.png"))

    total_frames = len(front_files)
    print(f"[2/2] Converting {total_frames} frames to YOLO-OBB labels...")
    for i, cpath in enumerate(front_files, 1):
        if i % 50 == 0 or i == total_frames:
            print(f"  converted {i}/{total_frames}")
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
            continue

        h, w = ids.shape[:2]
        uniq, counts = np.unique(ids, return_counts=True)

        lines = []
        for obj_id, count in zip(uniq, counts):
            obj_id = int(obj_id)
            if obj_id in (0, 65534):
                continue
            if count < args.min_pixels:
                continue
            if obj_id not in id2idx:
                continue
            mask = ids == obj_id
            line = obb_line_from_mask(
                mask, w, h, id2idx[obj_id], axis_aligned=obj_id in axis_aligned_ids
            )
            if line is not None:
                lines.append(line)

        label_path = out_dir / "labels" / (stem + ".txt")
        label_path.write_text("\n".join(lines), encoding="utf-8")

        try:
            shutil.copy2(cpath, out_dir / "images" / cpath.name)
        except Exception as e:
            logging.debug("Failed to copy image %s: %s", cpath, e)

    yaml = [
        f"path: {out_dir.resolve()}",
        "train: images",
        "val: images",
        f"names: {classes}",
    ]
    (out_dir / "data.yaml").write_text("\n".join(yaml), encoding="utf-8")
    print(f"Done. Exported to: {out_dir}")


if __name__ == "__main__":
    main()
