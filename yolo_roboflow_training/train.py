import os
from pathlib import Path
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
PROJECT_ID       = config["project_id"]
VERSION          = config["version"]
MODEL_TYPE       = config.get("model_type")
EPOCHS           = config.get("epochs")
IMGSZ            = config.get("imgsz")
BATCH            = config.get("batch")
RESULTS_DIR      = config.get("results_dir")
EXPORT_FORMATS   = config.get("export_formats")
DATASET_FORMAT   = config.get("dataset_format")

if not ROBOFLOW_API_KEY:
    raise ValueError("ROBOFLOW_API_KEY not set in environment or .env file.")


# Download dataset from Roboflow into 'roboflow_data/'
roboflow_data_dir = Path(__file__).parent / "roboflow_data"
roboflow_data_dir.mkdir(exist_ok=True)
rf       = Roboflow(api_key=ROBOFLOW_API_KEY)
project  = rf.workspace().project(PROJECT_ID)
versions = project.versions()
dataset  = project.version(VERSION).download(
    DATASET_FORMAT, location=str(roboflow_data_dir)
)

# Set up training configuration
model = YOLO(MODEL_TYPE)
subfolder_name = f"{dataset.name.replace(' ', '-')}-{VERSION}"
data_yaml_path = Path(dataset.location) / subfolder_name / "data.yaml"


# Select device: CUDA, MPS (Apple Silicon), or CPU
if torch.cuda.is_available():
    device = 0
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_built():
    device = 'mps'
    print("Using Apple Silicon MPS device.")
else:
    device = 'cpu'
    print("Using CPU.")

WORKERS = os.cpu_count() // 2 if os.cpu_count() else 2
print(f"Using {WORKERS} workers")

model.train(
    data=str(data_yaml_path),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=device,
    project=str(RESULTS_DIR),
    name="custom_yolov8",
    workers=WORKERS,
    # compile=True
)


# Evaluate the model
metrics = model.val(data=str(data_yaml_path))
print(f"Validation metrics: {metrics}")


# Export the trained model
for fmt in EXPORT_FORMATS:
    model.export(format=fmt, device=device)
