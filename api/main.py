
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import json
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image

from src.model import get_model, run_detection, image_to_bytes, bytes_to_image

app = FastAPI(
    title='YOLOv8 Object Detection API',
    description="""
A production-ready REST API for real-time object detection powered by
**YOLOv8** (Ultralytics). Detects 80 object classes from COCO dataset.

## Endpoints
- **POST /detect** — Upload an image, receive JSON detections
- **POST /detect/annotated** — Upload an image, receive annotated image
- **GET /classes** — List all 80 detectable classes

## How to use
Send a POST request to `/detect` with an image file.
Get back bounding box coordinates, class labels, and confidence scores.
    """,
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/', tags=['Health'])
async def root():
    return {
        'status':  'running',
        'model':   'YOLOv8 Nano',
        'classes': 80,
        'version': '1.0.0'
    }


@app.get('/health', tags=['Health'])
async def health():
    return {
        'status':  'healthy',
        'model':   'YOLOv8 Nano (yolov8n.pt)',
        'classes': 80,
        'version': '1.0.0'
    }


@app.get('/classes', tags=['Info'])
async def get_classes():
    model   = get_model()
    classes = {int(k): v for k, v in model.names.items()}
    return {'classes': classes, 'total': len(classes)}


@app.post('/detect', tags=['Detection'])
async def detect(
    file:       UploadFile = File(..., description="Image file (JPG, PNG, BMP)"),
    confidence: float      = Query(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Minimum confidence threshold (0.01 to 1.0)"
    )
):

    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Received: {file.content_type}"
        )

    try:
        image_bytes = await file.read()
        image       = bytes_to_image(image_bytes)
        result      = run_detection(image, confidence=confidence)

        return JSONResponse({
            'detection_count': result['detection_count'],
            'detections':      result['detections'],
            'class_summary':   result['class_summary'],
            'image_size':      {'width': image.width, 'height': image.height}
        })

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/detect/annotated', tags=['Detection'])
async def detect_annotated(
    file:       UploadFile = File(..., description="Image file (JPG, PNG, BMP)"),
    confidence: float      = Query(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Minimum confidence threshold"
    )
):

    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Received: {file.content_type}"
        )

    try:
        image_bytes     = await file.read()
        image           = bytes_to_image(image_bytes)
        result          = run_detection(image, confidence=confidence)
        annotated_bytes = image_to_bytes(result['annotated_image'])

        return Response(
            content=annotated_bytes,
            media_type='image/jpeg',
            headers={
                'X-Detection-Count': str(result['detection_count']),
                'X-Class-Summary':   json.dumps(result['class_summary'])
            }
        )

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)