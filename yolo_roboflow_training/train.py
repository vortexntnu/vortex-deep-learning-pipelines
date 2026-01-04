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
    raise RuntimeError(
        "ROBOFLOW_API_KEY must be set as an environment variable"
    ) from e
WORKSPACE_NAME = "hei-qp1ee"
PROJECT_NAME = "simulatorvalve-4rcbu"
VERSION = "1" 

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
versions = project.versions()
dataset = project.version(VERSION).download("yolov8")

# Use a pretrained YOLOv8m model to avoid training from scratch.
# TODO: Test different YOLOv8 model sizes.
model = YOLO(
    "yolov8m.pt"
)

data_yaml_path = os.path.join(dataset.location, "data.yaml")
results_dir = "results"

model.train(
    data=data_yaml_path,
    epochs=200,
    imgsz=640,
    batch=16,
    device=0,
    project=results_dir,
    name="custom_yolov8",
)

metrics = model.val(data=data_yaml_path)
print("Validation metrics:", metrics)

# Export formats for deployment
export_formats = ["onnx"]  
for fmt in export_formats:
    model.export(format=fmt, device=0)
