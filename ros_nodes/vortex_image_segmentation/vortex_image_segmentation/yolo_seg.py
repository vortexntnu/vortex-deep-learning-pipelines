"""YOLO segmentation model wrapper for inference and visualization using Ultralytics.

Defines parameter dataclass and main segmentation class.
"""

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


@dataclass
class YoloSegmentationParams:
    """Dataclass for storing YOLO segmentation parameters."""

    model_path: str
    device: str
    confidence_threshold: float
    max_detections: int
    imgsz: int
    compile: bool


class YoloSegmentation:
    """Wrapper class for YOLO segmentation model inference and visualization."""

    def __init__(self, params: YoloSegmentationParams) -> None:
        """Initialize the YOLO segmentation model with given parameters.

        Args:
            params (YoloSegmentationParams): Parameters for YOLO segmentation.
        """
        self._params: YoloSegmentationParams = params
        self._model: YOLO = self._load_model()

    def _load_model(self) -> YOLO:
        """Load the YOLO segmentation model.

        Returns:
            YOLO: Ultralytics YOLO model instance.
        """
        return YOLO(self._params.model_path, task="segment")

    def predict(self, cv_image: np.ndarray) -> list[Results]:
        """Run prediction on an input image and return the Results object(s).

        Args:
            cv_image (np.ndarray): Input image in OpenCV format.

        Returns:
            list[Results]: Ultralytics Results object(s) for the prediction.
        """
        results = self._model.predict(
            source=cv_image,
            imgsz=self._params.imgsz,
            conf=self._params.confidence_threshold,
            device=torch.device(self._params.device),
            max_det=self._params.max_detections,
            compile=self._params.compile,
        )
        return results

    def visualize(self, result: Results) -> np.ndarray:
        """Generate a visualization image from the Results object.

        Args:
            result (Results): Ultralytics Results object from prediction.

        Returns:
            np.ndarray: Visualization image with segmentation overlays.
        """
        return result.plot()
