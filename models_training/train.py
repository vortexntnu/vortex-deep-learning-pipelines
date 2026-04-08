#!/usr/bin/env python3

import argparse
import os
from datetime import datetime
from pathlib import Path

import cv2
import torch
import yaml
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

load_dotenv()

# Raw image dimensions coming out of Roboflow. Images get padded up to the
# next multiple of `PAD_MULTIPLE` so that every spatial dim is divisible by
# the network stride. Padding is applied to left/top/right only (never the
# bottom) per the dataset convention.
ORIG_H = 796
ORIG_W = 1496
PAD_MULTIPLE = 32
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def download_dataset(rf_cfg):
    base = Path("roboflow_data") / f"{rf_cfg['project_id']}-v{rf_cfg['version']}"
    base.mkdir(parents=True, exist_ok=True)

    if (base / "data.yaml").exists():
        print(f"Dataset already downloaded at {base}")
        return base / "data.yaml"

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    proj = rf.workspace(rf_cfg["workspace_id"]).project(rf_cfg["project_id"])

    dataset = proj.version(rf_cfg["version"]).download(
        rf_cfg["dataset_format"],
        location=str(base),
        overwrite=True,
    )

    return Path(dataset.location) / "data.yaml"


def compute_padding(h, w, multiple=PAD_MULTIPLE):
    """Pad (h, w) up to the nearest multiple. Padding goes on left/top/right,
    never the bottom. Width padding is split evenly between left and right
    (right gets the extra pixel if the total is odd)."""
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    top = new_h - h
    bottom = 0
    left = (new_w - w) // 2
    right = (new_w - w) - left
    return top, bottom, left, right, new_h, new_w


PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT, PADDED_H, PADDED_W = compute_padding(
    ORIG_H, ORIG_W
)


def _rewrite_label(label_path, old_w, old_h, new_w, new_h, left, top):
    """Shift normalized YOLO coordinates to account for padding. Handles both
    detection (cls xc yc w h) and polygon formats (cls x1 y1 x2 y2 ...) used
    for segmentation / OBB."""
    if not label_path.exists():
        return
    lines_out = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls, coords = parts[0], [float(v) for v in parts[1:]]
        if len(coords) == 4:
            xc, yc, bw, bh = coords
            new_coords = [
                (xc * old_w + left) / new_w,
                (yc * old_h + top) / new_h,
                (bw * old_w) / new_w,
                (bh * old_h) / new_h,
            ]
        else:
            new_coords = []
            for i in range(0, len(coords) - 1, 2):
                new_coords.append((coords[i] * old_w + left) / new_w)
                new_coords.append((coords[i + 1] * old_h + top) / new_h)
        lines_out.append(cls + " " + " ".join(f"{v:.6f}" for v in new_coords))
    label_path.write_text("\n".join(lines_out) + "\n")


def _preprocess_split(img_dir):
    """Convert every image in `img_dir` to single-channel grayscale uint8 and
    pad it. Corresponding labels in the sibling `labels/` dir are rewritten."""
    label_dir = img_dir.parent / "labels"
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue
        h, w = img.shape[:2]
        top, bottom, left, right, _, _ = compute_padding(h, w)
        padded = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
        cv2.imwrite(str(img_path), padded)
        _rewrite_label(
            label_dir / f"{img_path.stem}.txt",
            old_w=w,
            old_h=h,
            new_w=w + left + right,
            new_h=h + top + bottom,
            left=left,
            top=top,
        )


def preprocess_dataset(data_yaml_path):
    """Run grayscale conversion + padding over every split referenced by
    `data.yaml`. A sentinel file prevents re-processing on subsequent runs."""
    data_yaml_path = Path(data_yaml_path)
    base = data_yaml_path.parent
    marker = base / ".preprocessed_int8_gray"
    if marker.exists():
        print(f"Dataset already preprocessed at {base}")
        return

    cfg = yaml.safe_load(data_yaml_path.read_text())
    for key in ("train", "val", "valid", "test"):
        split = cfg.get(key)
        if not split:
            continue
        split_list = split if isinstance(split, list) else [split]
        for sp in split_list:
            img_dir = (base / sp).resolve()
            if img_dir.exists():
                print(f"Preprocessing {img_dir}")
                _preprocess_split(img_dir)

    marker.touch()


def fix_data_yaml(path):
    """Fix dataset paths in Roboflow YOLO `data.yaml`.

    Roboflow sometimes exports absolute/duplicated paths like
    `roboflow_data/project-v1/train/images`, but YOLO expects paths
    relative to `data.yaml`. This function rewrites them to
    `train/images`, `valid/images`, etc.
    """
    txt = Path(path).read_text()

    txt = txt.replace("roboflow_data/", "")
    txt = txt.replace("../", "")

    base = Path(path).parent.name
    txt = txt.replace(f"{base}/", "")

    Path(path).write_text(txt)


def train(config_path):
    config = load_config(config_path)

    task = config["run"]["task"]
    model_cfg = config["models"][task]
    rf_cfg = model_cfg["roboflow"]

    data_yaml = download_dataset(rf_cfg)
    fix_data_yaml(data_yaml)
    preprocess_dataset(data_yaml)

    model = YOLO(model_cfg["model"])
    device = get_device()

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_name = f"{task}-{timestamp}"

    # With rect=True Ultralytics keeps aspect ratio and sizes the long edge
    # to `imgsz`, so we pass the padded long edge here rather than the value
    # from config.yaml.
    imgsz = max(PADDED_H, PADDED_W)

    model.train(
        data=str(Path(data_yaml).resolve()),
        epochs=model_cfg["epochs"],
        imgsz=imgsz,
        batch=model_cfg["batch"],
        patience=model_cfg["patience"],
        device=device,
        project=config["run"]["output_dir"],
        name=run_name,
        rect=True,
    )

    model.val()

    export_formats = model_cfg.get("export", [])
    if export_formats:
        print(f"Exporting model to formats: {export_formats}")
        for fmt in export_formats:
            model.export(format=fmt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    train(args.config)
