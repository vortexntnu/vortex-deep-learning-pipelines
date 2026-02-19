#!/usr/bin/env python3
"""Train a YOLOv26 OBB object detection model using a Roboflow dataset.

This script is written to work with YOLOv2.6 oriented-bounding-box (OBB)
model names such as `yolo26n-obb`. It downloads a Roboflow dataset,
trains a model using the `ultralytics` API and exports the trained model.
"""

import os
from pathlib import Path

import torch
from roboflow import Roboflow
from ultralytics import YOLO

ROBOFLOW_WORKSPACE_NAME = "hei-qp1ee"
ROBOFLOW_PROJECT = "valve_obb_annotations-ejsa6"
ROBOFLOW_PROJECT_VERSION = "2"

# Use a pretrained YOLOv26n model to avoid training from scratch.
MODEL_NAME = "yolo26m-obb.pt"

# Specify the GPU device index to use for training.
DEVICE = 0


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")

    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("Using GPU:", torch.cuda.get_device_name(0))

    # The Roboflow API key is loaded from the environment to avoid hard-coding secrets.
    try:
        roboflow_api_key = os.environ["ROBOFLOW_API_KEY"]
    except KeyError as e:
        raise RuntimeError(
            "ROBOFLOW_API_KEY must be set as an environment variable"
        ) from e

    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE_NAME).project(ROBOFLOW_PROJECT)

    # Export format: Roboflow's "yolov8" export works well with ultralytics
    # and also with recent YOLOv2.6 OBB support when datasets include OBB
    dataset = project.version(ROBOFLOW_PROJECT_VERSION).download("yolov8")
    data_yaml_path = Path(dataset.location) / "data.yaml"

    experiment_name = f"{ROBOFLOW_PROJECT}_v{ROBOFLOW_PROJECT_VERSION}_{MODEL_NAME}"

    # Resolve model argument for ultralytics YOLO: allow either a local .pt
    # or a model name like 'yolo26n-obb' which the library may resolve.
    model_arg = MODEL_NAME
    if not MODEL_NAME.endswith(".pt") and not Path(f"{MODEL_NAME}.pt").exists():
        model_arg = MODEL_NAME
    elif Path(MODEL_NAME).exists() or Path(f"{MODEL_NAME}.pt").exists():
        model_arg = str(
            Path(MODEL_NAME)
            if Path(MODEL_NAME).suffix == ".pt"
            else Path(f"{MODEL_NAME}.pt")
        )

    model = YOLO(model_arg)

    device_str = f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu"

    results = model.train(
        data=str(data_yaml_path),
        epochs=200,  # Bump up the number of epochs if necessary
        imgsz=640,
        batch=16,
        device=device_str,
        workers=8,
        project=Path("results") / "yolov26_obb",
        name=experiment_name,
    )

    save_dir = Path(results.save_dir)
    best_weights_path = save_dir / "weights" / "best.pt"

    print("Ultralytics saved this run to:", save_dir)
    print("Looking for best checkpoint at:", best_weights_path)

    if not best_weights_path.exists():
        raise FileNotFoundError(
            f"Trained model checkpoint not found at {best_weights_path}. "
            "Ensure training completed successfully before validation and export."
        )

    trained_model = YOLO(str(best_weights_path))

    metrics = trained_model.val(data=str(data_yaml_path))
    print("Validation metrics:", metrics)

    # Export formats for deployment
    for fmt in ["onnx"]:
        trained_model.export(format=fmt, device=DEVICE)


if __name__ == "__main__":
    main()
