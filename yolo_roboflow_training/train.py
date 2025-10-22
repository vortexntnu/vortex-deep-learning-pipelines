import os
import yaml
from dotenv import load_dotenv
import torch
from ultralytics import YOLO
from roboflow import Roboflow

# Load environment variables from .env file
load_dotenv()

# Print CUDA info
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch cuDNN version: {torch.backends.cudnn.version()}")

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Parameters
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
PROJECT_NAME = config["project_name"]
VERSION = config["version"]
MODEL_TYPE = config.get("model_type", "yolov8m.pt")
EPOCHS = config.get("epochs", 200)
IMGSZ = config.get("imgsz", 640)
BATCH = config.get("batch", 16)
RESULTS_DIR = config.get("results_dir", "results")
EXPORT_FORMATS = config.get("export_formats", ["onnx"])

if not ROBOFLOW_API_KEY:
    raise ValueError("ROBOFLOW_API_KEY not set in environment or .env file.")

# Download dataset from Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(PROJECT_NAME)
versions = project.versions()
# print(f"Available versions: {[v['id'] for v in versions]}")
dataset = project.version(VERSION).download("yolov8")

# Set up training configuration
model = YOLO(MODEL_TYPE)
data_yaml_path = os.path.join(dataset.location, "data.yaml")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Ensure your environment supports GPU acceleration.")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Train the YOLOv8 model
model.train(
    data=data_yaml_path,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=0,
    project=RESULTS_DIR,
    name="custom_yolov8"
)

# Evaluate the model
metrics = model.val(data=data_yaml_path)
print(f"Validation metrics: {metrics}")

# Export the trained model
for fmt in EXPORT_FORMATS:
    model.export(format=fmt, device=0)
