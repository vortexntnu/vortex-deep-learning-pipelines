"""The main entry point for YOLO detector buoys."""

from os import getcwd, path

from ultralytics import YOLO
from utils import get_data, get_device, process_video

if "__main__" == __name__:
    device = get_device()
    model = YOLO("yolov8n.pt")

    file_path = path.abspath(getcwd())
    dataset = get_data()

    result = model.train(
        data=dataset.location + "\\data.yaml", epochs=50, imgsz=640, device=device
    )

    model.val()

    process_video("https://youtu.be/4WGpIOwkLA4?feature=shared", model)

# References:
# https://docs.ultralytics.com/quickstart/#install-ultralytics
