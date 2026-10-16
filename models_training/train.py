#!/usr/bin/env python3

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(framework):
    if torch.cuda.is_available():
        return "cuda" if framework == "rf-detr" else 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_framework(config):
    framework = config["run"].get("framework", "yolo").lower().replace("_", "-")
    if framework not in {"yolo", "rf-detr"}:
        raise ValueError(
            f"Unknown framework '{framework}'. Expected 'yolo' or 'rf-detr'."
        )
    return framework


def download_dataset(rf_cfg):
    base = Path("roboflow_data") / f"{rf_cfg['project_id']}-v{rf_cfg['version']}"
    base.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    proj = rf.workspace(rf_cfg["workspace_id"]).project(rf_cfg["project_id"])

    dataset = proj.version(rf_cfg["version"]).download(
        rf_cfg["dataset_format"],
        location=str(base),
        overwrite=True,
    )

    return Path(dataset.location)


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


def get_data_path(dataset_path, task):
    if task == "classify":
        return dataset_path

    data_path = dataset_path / "data.yaml"
    if not data_path.is_file():
        raise FileNotFoundError(f"Roboflow dataset YAML not found: {data_path}")

    fix_data_yaml(data_path)
    return data_path


def train_yolo(config, task, model_cfg, dataset_path, run_name):
    # Import only the selected framework so RF-DETR can be used independently.
    from ultralytics import YOLO

    model = YOLO(model_cfg["model"], task=task)
    data_path = get_data_path(dataset_path, task)

    model.train(
        data=str(data_path.resolve()),
        task=task,
        epochs=model_cfg["epochs"],
        imgsz=model_cfg["imgsz"],
        batch=model_cfg["batch"],
        patience=model_cfg["patience"],
        weight_decay=model_cfg.get("weight_decay", 0.0005),
        device=get_device("yolo"),
        project=config["run"]["output_dir"],
        name=run_name,
    )

    model.val()

    export_formats = model_cfg.get("export", [])
    if export_formats:
        print(f"Exporting model to formats: {export_formats}")
        for fmt in export_formats:
            model.export(format=fmt)


def train_rf_detr(config, task, model_cfg, dataset_path, run_name):
    if task not in {"detect", "segment"}:
        raise ValueError(
            f"RF-DETR does not support the '{task}' task in this pipeline. "
            "Use 'detect' or 'segment', or set run.framework to 'yolo'."
        )

    rf_detr_cfg = model_cfg.get("rf_detr")
    if not rf_detr_cfg:
        raise ValueError(
            f"No models.{task}.rf_detr configuration exists for RF-DETR training."
        )

    import rfdetr

    model_class_name = rf_detr_cfg["model_class"]
    allowed_models = {
        "detect": {"RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge"},
        "segment": {
            "RFDETRSegNano",
            "RFDETRSegSmall",
            "RFDETRSegMedium",
            "RFDETRSegLarge",
        },
    }
    if model_class_name not in allowed_models[task]:
        choices = ", ".join(sorted(allowed_models[task]))
        raise ValueError(
            f"Invalid RF-DETR model_class '{model_class_name}' for {task}. "
            f"Choose one of: {choices}."
        )

    model_class = getattr(rfdetr, model_class_name)
    model = model_class(
        gradient_checkpointing=rf_detr_cfg.get("gradient_checkpointing", False)
    )
    output_dir = Path(config["run"]["output_dir"]) / run_name

    data_yaml = dataset_path / "data.yaml"
    if data_yaml.is_file():
        fix_data_yaml(data_yaml)

    model.train(
        dataset_dir=str(dataset_path.resolve()),
        output_dir=str(output_dir.resolve()),
        epochs=rf_detr_cfg["epochs"],
        batch_size=rf_detr_cfg["batch"],
        grad_accum_steps=rf_detr_cfg.get("grad_accum_steps", 1),
        lr=rf_detr_cfg.get("lr", 1e-4),
        resolution=rf_detr_cfg["resolution"],
        weight_decay=rf_detr_cfg.get("weight_decay", 1e-4),
        device=get_device("rf-detr"),
        early_stopping=rf_detr_cfg.get("early_stopping", True),
        early_stopping_patience=rf_detr_cfg.get("patience", 10),
        multi_scale=rf_detr_cfg.get("multi_scale", "per-batch"),
    )

    model.evaluate(split="val")

    export_formats = rf_detr_cfg.get("export", [])
    if export_formats:
        print(f"Exporting model to formats: {export_formats}")
        for fmt in export_formats:
            model.export(format=fmt, output_dir=str(output_dir.resolve()))


def train(config_path):
    config = load_config(config_path)
    framework = get_framework(config)
    task = config["run"]["task"]
    model_cfg = config["models"][task]

    if framework == "rf-detr" and (
        task not in {"detect", "segment"} or "rf_detr" not in model_cfg
    ):
        raise ValueError(
            f"Task '{task}' has no RF-DETR configuration. "
            "Use detect/segment with a models.<task>.rf_detr section, or use YOLO."
        )

    dataset_path = download_dataset(model_cfg["roboflow"])

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_name = f"{task}-{framework}-{timestamp}"

    if framework == "yolo":
        train_yolo(config, task, model_cfg, dataset_path, run_name)
    else:
        train_rf_detr(config, task, model_cfg, dataset_path, run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    train(args.config)
