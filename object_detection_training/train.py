#!/usr/bin/env python3

import os

import torch
from roboflow import Roboflow
from ultralytics import YOLO

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required but not available.")

print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("Using GPU:", torch.cuda.get_device_name(0))

# Roboflow dataset configuration.
# The API key is loaded from the environment to avoid hard-coding secrets.
try:
    ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]
except KeyError as e:
    raise RuntimeError("ROBOFLOW_API_KEY must be set as an environment variable") from e
ROBOFLOW_WORKSPACE_NAME = "hei-qp1ee"
ROBOFLOW_PROJECT_NAME = "simulatorvalve-4rcbu"
ROBOFLOW_PROJECT_VERSION = "1"

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE_NAME).project(ROBOFLOW_PROJECT_NAME)
versions = project.versions()
dataset = project.version(ROBOFLOW_PROJECT_VERSION).download("yolov8")

# Use a pretrained YOLOv8m model to avoid training from scratch.
# TODO: Test different YOLOv8 model sizes.
MODEL_NAME = "yolov8m"
model = YOLO(f"{MODEL_NAME}.pt")

data_yaml_path = os.path.join(dataset.location, "data.yaml")
results_dir = "results/yolov8"
experiment_name = f"{ROBOFLOW_PROJECT_NAME}_v{ROBOFLOW_PROJECT_VERSION}_{MODEL_NAME}"

model.train(
    data=data_yaml_path,
    epochs=5,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    project=results_dir,
    name=experiment_name,
)

metrics = model.val(data=data_yaml_path)
print("Validation metrics:", metrics)

# Export formats for deployment
export_formats = ["onnx"]
for fmt in export_formats:
    model.export(format=fmt, device=0)
