import os

# !which python
# !pip show ultralytics
# !pip show urllib3
# !pip show requests
# !pip show requests-toolbelt
# Check GPU availability
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("PyTorch built with CUDA:", torch.backends.cudnn.version())


from roboflow import Roboflow
from ultralytics import YOLO

# Step 1: Download Dataset from Roboflow
ROBOFLOW_API_KEY = ""  # Replace with your Roboflow API Key
PROJECT_NAME = ""  # Replace with your project id
VERSION = ""  # Replace with your dataset version number exluding the 'v'

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(PROJECT_NAME)
# List all available versions
versions = project.versions()
# print("Available versions:", [v['id'] for v in versions])
dataset = project.version(VERSION).download("yolov8")


# Step 2: Set up training configuration
model = YOLO(
    "yolov8m.pt"
)  # Use the smallest YOLOv8 model to start. Change to 'yolov8s.pt', etc., for larger models.

# Define paths
data_yaml_path = os.path.join(
    dataset.location, "data.yaml"
)  # Path to the dataset's data.yaml file
results_dir = "results"  # Directory to save training results


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Ensure your environment supports GPU acceleration."
    )

print("Using GPU:", torch.cuda.get_device_name(0))

# Step 3: Train the YOLOv8 model
model.train(
    data=data_yaml_path,  # Path to dataset YAML file
    epochs=200,  # Number of training epochs
    imgsz=640,  # Image size
    batch=16,  # Batch size
    device=0,  # Use the first GPU (0). For multiple GPUs, use device="0,1,2"
    project=results_dir,  # Directory for saving results
    name="custom_yolov8",  # Subdirectory for this training run
)

# Step 4: Evaluate the model
metrics = model.val(data=data_yaml_path)
print("Validation metrics:", metrics)

# Step 5: Export the trained model
export_formats = ["onnx"]  # Export formats for deployment
for fmt in export_formats:
    model.export(format=fmt, device=0)  # Ensure GPU is used during export if applicable
