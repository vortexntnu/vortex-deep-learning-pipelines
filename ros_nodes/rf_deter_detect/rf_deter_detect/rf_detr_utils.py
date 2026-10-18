#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str


class OnnxPredictions:
    def __init__(self, boxes, scores, class_ids, class_names):
        self.xyxy = boxes
        self.confidence = scores
        self.class_id = class_ids
        self.data = {'class_name': class_names}


class OnnxRfDetrModel:
    """Small ONNX Runtime adapter exposing the same fields as RF-DETR predict."""

    def __init__(
        self,
        model_path: str,
        device: str,
        class_names: list[str],
        background_class_id: int = -1,
    ):
        import onnxruntime as ort

        normalized_device = normalize_device(device)
        if normalized_device.startswith('cuda') and hasattr(ort, 'preload_dlls'):
            # Load CUDA 12 and cuDNN 9 installed by the onnxruntime-gpu extras.
            ort.preload_dlls(directory='')
        available = ort.get_available_providers()
        if normalized_device == 'cpu':
            providers = ['CPUExecutionProvider']
        elif normalized_device.startswith('cuda'):
            if 'CUDAExecutionProvider' not in available:
                from importlib import metadata

                installed_runtimes = []
                for package_name in ('onnxruntime', 'onnxruntime-gpu'):
                    try:
                        version = metadata.version(package_name)
                        installed_runtimes.append(f'{package_name}=={version}')
                    except metadata.PackageNotFoundError:
                        pass
                installed = ', '.join(installed_runtimes) or 'none'
                raise RuntimeError(
                    'CUDA was requested, but ONNX Runtime has no CUDA provider. '
                    f'Installed runtime packages: {installed}. If both CPU and GPU '
                    'packages are listed, uninstall both and reinstall only '
                    "'onnxruntime-gpu[cuda,cudnn]'. Otherwise launch with "
                    "device:='cpu'."
                )
            device_id = 0
            if ':' in normalized_device:
                device_id = int(normalized_device.split(':', maxsplit=1)[1])
            providers = [
                (
                    'CUDAExecutionProvider',
                    {
                        'device_id': device_id,
                        'cudnn_conv_use_max_workspace': '1',
                        'prefer_nhwc': '1',
                    },
                ),
                'CPUExecutionProvider',
            ]
        else:
            raise ValueError(
                f"ONNX inference does not support device '{device}'. "
                "Use 'cpu', 'cuda', 'cuda:N', or a numeric GPU index."
            )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # FP16 graphs can emit many harmless constant-folding warnings at startup.
        session_options.log_severity_level = 3
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers,
        )
        model_input = self.session.get_inputs()[0]
        if len(model_input.shape) != 4:
            raise ValueError(
                f'Expected a 4D RF-DETR ONNX input, got {model_input.shape}'
            )
        _, channels, height, width = model_input.shape
        if not all(isinstance(value, int) for value in (channels, height, width)):
            raise ValueError(
                f'RF-DETR ONNX input must have a fixed shape, got {model_input.shape}'
            )

        self.input_name = model_input.name
        self.input_height = height
        self.input_width = width
        self.input_dtype = (
            np.float16 if model_input.type == 'tensor(float16)' else np.float32
        )
        self.class_names = class_names
        self.background_class_id = background_class_id
        self._mean_255 = tuple(
            float(value * 255.0) for value in (0.485, 0.456, 0.406)
        )
        self._inverse_std = np.reciprocal(
            np.array([0.229, 0.224, 0.225], dtype=np.float32)
        ).reshape(1, 3, 1, 1)

    def _preprocess(self, frame, swap_red_blue=False):
        tensor = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(self.input_width, self.input_height),
            mean=self._mean_255,
            swapRB=swap_red_blue,
            crop=False,
        )
        tensor *= self._inverse_std
        return np.asarray(tensor, dtype=self.input_dtype, order='C')

    def _without_background(self, scores):
        class_ids = np.arange(scores.shape[1], dtype=np.int64)
        background_id = self.background_class_id
        if background_id < 0:
            background_id += scores.shape[1]
        if 0 <= background_id < scores.shape[1]:
            keep = class_ids != background_id
            return scores[:, keep], class_ids[keep]
        return scores, class_ids

    def _predict(self, frame, threshold, swap_red_blue):
        input_tensor = self._preprocess(frame, swap_red_blue=swap_red_blue)
        outputs = {
            output.name: value
            for output, value in zip(
                self.session.get_outputs(),
                self.session.run(None, {self.input_name: input_tensor}),
            )
        }
        try:
            boxes_cxcywh = outputs['dets'][0]
            logits = outputs['labels'][0]
        except KeyError as error:
            raise ValueError(
                "RF-DETR ONNX outputs must be named 'dets' and 'labels'"
            ) from error

        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -88.0, 88.0)))
        scores, exported_class_ids = self._without_background(scores)
        flat_scores = scores.reshape(-1)
        selection_count = min(boxes_cxcywh.shape[0], flat_scores.size)
        if selection_count == 0:
            selected = np.empty(0, dtype=np.int64)
        else:
            selected = np.argpartition(flat_scores, -selection_count)[
                -selection_count:
            ]
            selected = selected[np.argsort(flat_scores[selected])[::-1]]
            selected = selected[flat_scores[selected] > threshold]

        query_ids, class_slots = np.divmod(selected, scores.shape[1])
        class_ids = exported_class_ids[class_slots]
        selected_boxes = boxes_cxcywh[query_ids]

        image_height, image_width = frame.shape[:2]
        cx, cy, box_width, box_height = selected_boxes.T
        boxes = np.stack(
            [
                cx - box_width / 2.0,
                cy - box_height / 2.0,
                cx + box_width / 2.0,
                cy + box_height / 2.0,
            ],
            axis=1,
        )
        scale = np.array(
            [image_width, image_height, image_width, image_height],
            dtype=np.float32,
        )
        boxes = np.clip(boxes * scale, 0.0, scale)
        names = np.array(
            [
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else str(class_id)
                for class_id in class_ids
            ]
        )
        return OnnxPredictions(
            boxes,
            flat_scores[selected],
            class_ids,
            names,
        )

    def predict(self, rgb_frame, threshold=0.5, include_source_image=False):
        del include_source_image
        return self._predict(rgb_frame, threshold, swap_red_blue=False)

    def predict_bgr(self, bgr_frame, threshold=0.5):
        return self._predict(bgr_frame, threshold, swap_red_blue=True)


def normalize_device(device: str) -> str:
    """Translate YOLO-style numeric GPU identifiers to PyTorch device names."""
    value = str(device).strip().lower()
    if value.isdigit():
        return f'cuda:{value}'
    return value


def load_model(
    model_path: str,
    device: str,
    trust_checkpoint: bool = False,
    class_names: list[str] | None = None,
    background_class_id: int = -1,
):
    """Load either an exported ONNX model or a trained RF-DETR checkpoint."""
    if Path(model_path).suffix.lower() == '.onnx':
        return OnnxRfDetrModel(
            model_path,
            device,
            class_names or [],
            background_class_id,
        )

    from rfdetr import RFDETR

    return RFDETR.from_checkpoint(
        model_path,
        device=normalize_device(device),
        trust_checkpoint=trust_checkpoint,
    )


def _class_names(predictions, count: int) -> list[str]:
    data = getattr(predictions, 'data', {}) or {}
    names = data.get('class_name')
    if names is None:
        return [''] * count
    return [str(name) for name in names]


def process_frame(frame, model, conf: float, annotate: bool = True):
    """Run RF-DETR inference on a BGR OpenCV frame and draw its detections."""
    if hasattr(model, 'predict_bgr'):
        predictions = model.predict_bgr(frame, threshold=conf)
    else:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = model.predict(
            rgb_frame,
            threshold=conf,
            include_source_image=False,
        )

    boxes = np.asarray(predictions.xyxy)
    scores = np.asarray(predictions.confidence)
    class_ids = np.asarray(predictions.class_id)
    class_names = _class_names(predictions, len(boxes))

    detections = []
    annotated = frame.copy() if annotate else None

    for box, score, class_id, class_name in zip(
        boxes, scores, class_ids, class_names
    ):
        x1, y1, x2, y2 = (float(value) for value in box)
        score = float(score)
        class_id = int(class_id)
        label_name = class_name or str(class_id)

        detections.append(
            Detection(x1, y1, x2, y2, score, class_id, label_name)
        )

        if annotated is not None:
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f'{label_name} {score:.2f}',
                (p1[0], max(p1[1] - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    return detections, annotated
