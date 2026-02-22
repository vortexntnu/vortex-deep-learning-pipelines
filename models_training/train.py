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
        return base / "data.yaml"

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    proj = rf.workspace(rf_cfg["workspace_id"]).project(rf_cfg["project_id"])

    dataset = proj.version(rf_cfg["version"]).download(
        rf_cfg["dataset_format"],
        location=str(base),
        overwrite=True,
    )

    return Path(dataset.location) / "data.yaml"


def fix_data_yaml(path):
    """Fix incorrect dataset paths inside a Roboflow-exported YOLO `data.yaml`.

    Why this is needed
    ------------------
    Roboflow sometimes exports `data.yaml` with paths that include the dataset
    folder name (e.g. `roboflow_data/project-v1/train/images`). However, YOLO
    interprets paths relative to the directory containing `data.yaml`. If the
    dataset is already inside `roboflow_data/project-v1/`, this results in
    duplicated paths like:

        roboflow_data/project-v1/roboflow_data/project-v1/train/images

    which do not exist, causing training to fail with "images not found".

    What this function does
    -----------------------
    It rewrites the `data.yaml` file so dataset paths become relative, e.g.:

        roboflow_data/project-v1/train/images  ->  train/images
        ../test/images                         ->  test/images

    This matches YOLO's expected directory structure and prevents dataset
    path errors without needing to manually edit the exported dataset.

    Notes:
    -----
    - Safe to run multiple times (idempotent).
    - Only modifies path strings inside `data.yaml`.
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

    model = YOLO(model_cfg["model"])
    device = get_device()

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_name = f"{task}-{timestamp}"

    model.train(
        data=str(Path(data_yaml).resolve()),
        epochs=model_cfg["epochs"],
        imgsz=model_cfg["imgsz"],
        batch=model_cfg["batch"],
        patience=model_cfg["patience"],
        device=device,
        project=config["run"]["output_dir"],
        name=run_name,
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
