
import os
import io
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'yolov8n.pt')

DEFAULT_CONFIDENCE = 0.25

_model = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        print(f"Loading YOLOv8 model...")

        if os.path.exists(MODEL_PATH):
            _model = YOLO(MODEL_PATH)
        elif os.path.exists('yolov8n.pt'):
            import shutil
            shutil.move('yolov8n.pt', MODEL_PATH)
            _model = YOLO(MODEL_PATH)
        else:
            _model = YOLO('yolov8n.pt')

        print(f"  Model loaded. Classes: {len(_model.names)}")
    return _model


def run_detection(image: Image.Image, confidence: float = DEFAULT_CONFIDENCE) -> dict:

    model = get_model()

    results = model(image, conf=confidence, verbose=False)

    result = results[0]

    detections = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes      = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids  = result.boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            class_name = model.names[int(cls_id)]
            detections.append({
                'class':      class_name,
                'confidence': round(float(conf), 4),
                'bbox': {
                    'x1': int(box[0]),
                    'y1': int(box[1]),
                    'x2': int(box[2]),
                    'y2': int(box[3]),
                }
            })


    annotated_array = result.plot()

    annotated_rgb   = annotated_array[:, :, ::-1]
    annotated_image = Image.fromarray(annotated_rgb)

    class_summary = {}
    for det in detections:
        cls = det['class']
        class_summary[cls] = class_summary.get(cls, 0) + 1

    return {
        'annotated_image': annotated_image,
        'detections':      detections,
        'detection_count': len(detections),
        'class_summary':   class_summary,
    }


def image_to_bytes(image: Image.Image, format: str = 'JPEG') -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def bytes_to_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert('RGB')
