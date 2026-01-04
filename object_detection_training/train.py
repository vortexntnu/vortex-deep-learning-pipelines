#!/usr/bin/env python3
"""
Train a YOLOv8 object detection model using a Roboflow dataset.
"""

import os
from pathlib import Path

import torch
from roboflow import Roboflow
from ultralytics import YOLO

ROBOFLOW_WORKSPACE_NAME = "hei-qp1ee"
ROBOFLOW_PROJECT_NAME = "simulatorvalve-4rcbu"
ROBOFLOW_PROJECT_VERSION = "1"

# Use a pretrained YOLOv8m model to avoid training from scratch. 
# TODO: Test different YOLOv8 model sizes.
MODEL_NAME = "yolov8m"

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
        raise RuntimeError("ROBOFLOW_API_KEY must be set as an environment variable") from e

    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE_NAME).project(ROBOFLOW_PROJECT_NAME)

    dataset = project.version(ROBOFLOW_PROJECT_VERSION).download("yolov8")
    data_yaml_path = Path(dataset.location) / "data.yaml"

    experiment_name = f"{ROBOFLOW_PROJECT_NAME}_v{ROBOFLOW_PROJECT_VERSION}_{MODEL_NAME}"

    model = YOLO(f"{MODEL_NAME}.pt")

    model.train(
        data=str(data_yaml_path),
        epochs=200,
        imgsz=640,
        batch=16,
        device=DEVICE,
        workers=8,
        project = Path("results") / "yolov8",
        name=experiment_name,
    )

    metrics = model.val(data=str(data_yaml_path))
    print("Validation metrics:", metrics)
    
    # Export formats for deployment
    for fmt in ["onnx"]:
        model.export(format=fmt, device=DEVICE)


if __name__ == "__main__":
    main()
