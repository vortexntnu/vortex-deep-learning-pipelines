#!/usr/bin/env python3

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

load_dotenv()


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
        return base

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    proj = rf.workspace(rf_cfg["workspace_id"]).project(rf_cfg["project_id"])

    dataset = proj.version(rf_cfg["version"]).download(
        rf_cfg["dataset_format"],
        location=str(base),
        overwrite=True,
    )

    return Path(dataset.location)


def fix_data_yaml(path):
    """Fix dataset paths in Roboflow YOLO `data.yaml` and handle missing splits.

    Roboflow sometimes exports absolute/duplicated paths like
    `roboflow_data/project-v1/train/images`, but YOLO expects paths
    relative to `data.yaml`. This function rewrites them to
    `train/images`, `valid/images`, etc.

    Also handles datasets exported without a val or test split: falls
    back to the train split for val (YOLO requires val to train), and
    drops the test entry entirely when the directory is missing.
    """
    path = Path(path)
    txt = path.read_text()

    txt = txt.replace("roboflow_data/", "")
    txt = txt.replace("../", "")
    txt = txt.replace(f"{path.parent.name}/", "")

    data = yaml.safe_load(txt)
    dataset_dir = path.parent

    val_rel = data.get("val")
    if not val_rel or not (dataset_dir / val_rel).exists():
        print(
            f"No val split found in {dataset_dir}, falling back to train split for validation"
        )
        data["val"] = data["train"]

    test_rel = data.get("test")
    if test_rel and not (dataset_dir / test_rel).exists():
        print(f"No test split found in {dataset_dir}, removing test entry")
        data.pop("test")

    path.write_text(yaml.safe_dump(data, sort_keys=False))


def normalize_obb_labels(dataset_path):
    """Promote axis-aligned YOLO labels to 8-coord OBB labels in-place.

    An OBB dataset in Roboflow can be annotated with either the
    polygon/rotated-box tool or the plain axis-aligned bbox tool. On
    export, the first becomes an 8-coord quad (valid OBB), but the
    second stays in 4-coord `cx cy w h` detect format in the same
    label file. YOLO's OBB trainer can't interpret those 4-coord
    lines: affected classes train with degenerate (zero-thickness)
    ground truth, show up as thin-line predictions, and get mAP ~0.

    An axis-aligned box is just an OBB with rotation = 0, so we
    rewrite each 4-coord line into its 4 corners (TL, TR, BR, BL).
    Lines with >=8 coords are left alone, so running this on an
    already-correct dataset is a no-op.
    """
    converted_files = 0
    converted_lines = 0

    for label_file in Path(dataset_path).rglob("labels/*.txt"):
        lines = label_file.read_text().splitlines()
        new_lines = []
        changed = False
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                new_lines.append(line)
                continue
            cls = parts[0]
            cx, cy, w, h = map(float, parts[1:])
            hw, hh = w / 2, h / 2
            corners = [
                cx - hw,
                cy - hh,
                cx + hw,
                cy - hh,
                cx + hw,
                cy + hh,
                cx - hw,
                cy + hh,
            ]
            new_lines.append(f"{cls} " + " ".join(f"{c:.6f}" for c in corners))
            changed = True
            converted_lines += 1
        if changed:
            label_file.write_text("\n".join(new_lines) + "\n")
            converted_files += 1

    for cache in Path(dataset_path).rglob("labels.cache"):
        cache.unlink()

    if converted_lines:
        print(
            f"Converted {converted_lines} axis-aligned labels to OBB format "
            f"across {converted_files} files"
        )


def has_test_split(task, dataset_path, data_arg):
    if task == "classify":
        return (Path(dataset_path) / "test").is_dir()

    data_cfg = yaml.safe_load(Path(data_arg).read_text())
    test_rel = data_cfg.get("test")
    if not test_rel:
        return False
    test_path = (Path(data_arg).parent / test_rel).resolve()
    return test_path.exists()


def train(config_path):
    config = load_config(config_path)

    task = config["run"]["task"]
    model_cfg = config["models"][task]
    rf_cfg = model_cfg["roboflow"]

    dataset_path = download_dataset(rf_cfg)

    # The reason we do this is because the classify task expects a directory of images, while the other tasks expect a data.yaml file
    if task == "classify":
        data_arg = str(dataset_path.resolve())
    else:
        data_yaml = dataset_path / "data.yaml"
        fix_data_yaml(data_yaml)
        if task == "obb":
            normalize_obb_labels(dataset_path)
        data_arg = str(data_yaml.resolve())

    model = YOLO(model_cfg["model"], task=task)
    device = get_device()

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_name = f"{task}-{timestamp}"

    model.train(
        data=data_arg,
        task=task,
        epochs=model_cfg["epochs"],
        imgsz=model_cfg["imgsz"],
        batch=model_cfg["batch"],
        patience=model_cfg["patience"],
        device=device,
        project=config["run"]["output_dir"],
        name=run_name,
    )

    model.val()

    if has_test_split(task, dataset_path, data_arg):
        print("Evaluating on test split")
        model.val(split="test")
    else:
        print("No test split found, skipping test evaluation")

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
