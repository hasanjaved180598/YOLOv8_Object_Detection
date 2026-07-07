# 🎯 YOLOv8 Real-Time Object Detection

A production-ready **object detection system** powered by **YOLOv8 Nano**, served through a **FastAPI** REST API and visualised with a **Streamlit** dashboard. Detects **80 object classes** from the COCO dataset including people, vehicles, animals, and everyday items — no training required.

---

## 📌 Table of Contents

- [What is Object Detection?](#-what-is-object-detection)
- [What is YOLO?](#-what-is-yolo)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Key Concepts](#-key-concepts)
- [API Endpoints](#-api-endpoints)
- [Detectable Classes](#-detectable-classes)
- [Real Debugging Journey](#-real-debugging-journey)
- [Demo](#-demo)
- [Setup and Installation](#-setup-and-installation)
- [How to Run](#-how-to-run)
- [Project Workflow](#-project-workflow)

---

## 🎯 What is Object Detection?

Object detection solves two problems simultaneously for every object in an image:

| Task | Question | Output |
|------|----------|--------|
| Classification | What is this object? | Class label (person, car, dog...) |
| Localisation | Where is it? | Bounding box coordinates (x1, y1, x2, y2) |

This is harder than image classification (which assigns one label to the whole image) because the model must find and identify every object present — including multiple instances of the same class.

**Applications:** Security surveillance, self-driving cars, retail shelf monitoring, delivery verification, medical imaging, sports analytics, robotics.

---

## ⚡ What is YOLO?

**YOLO** stands for **You Only Look Once**.

Older detection methods scanned images multiple times in overlapping windows — slow, sometimes seconds per image. YOLO's breakthrough was processing the entire image in a **single forward pass** by dividing it into a grid and predicting bounding boxes and class probabilities for all grid cells simultaneously.

**Result:** Fast enough for real-time video on GPU. Still usable on CPU for images.

**YOLOv8** (Ultralytics, 2023) is the latest version — pre-trained on **COCO dataset** with 118,000 images across 80 object classes.

### Model Size Options

| Model | Size | Use Case |
|-------|------|----------|
| **yolov8n (nano) ← we use this** | 6MB | CPU deployment, fast demo |
| yolov8s (small) | 22MB | Balanced speed/accuracy |
| yolov8m (medium) | 52MB | Production on CPU |
| yolov8l (large) | 87MB | GPU recommended |
| yolov8x (extra) | 131MB | Maximum accuracy |

---

## 📁 Project Structure

```
yolo_detector/
├── src/
│   └── model.py            ← YOLOv8 loading, inference, box drawing
├── api/
│   └── main.py             ← FastAPI with 5 endpoints
├── app/
│   └── app.py              ← Streamlit dashboard
├── models/
│   └── yolov8n.pt          ← model weights (auto-downloaded, ~6MB)
├── notebook/
│   └── exploration.ipynb   ← model testing and class exploration
└── requirements.txt
```

---

## 🛠 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.11 | Core language |
| Model | YOLOv8 Nano (Ultralytics) | Object detection |
| API Framework | FastAPI | REST API |
| ASGI Server | Uvicorn | Serves FastAPI |
| Image Processing | Pillow, NumPy | Image reading and conversion |
| Dashboard | Streamlit | Visual frontend |
| Charts | Plotly | Detection count bar chart |

---

## 🧠 Key Concepts

### Transfer Learning
Same concept as ResNet50 for the pneumonia detector. YOLOv8 was pre-trained on 118,000 COCO images — weeks of GPU time. We load those pre-trained weights and run inference directly — zero training needed. The model already knows what 80 object classes look like.

---

### Two Detection Endpoints — Why?

```
POST /detect           → returns JSON with coordinates and labels
POST /detect/annotated → returns image with boxes already drawn
```

Different clients have different needs:
- A **mobile app** wants raw JSON to draw its own custom UI
- A **Streamlit dashboard** wants a ready-to-display annotated image
- A **data pipeline** wants JSON to store detections in a database

Serving both means any client can use the same API without modification.

---

### BGR to RGB Conversion
YOLOv8 uses OpenCV internally which stores pixels in **BGR** order (Blue-Green-Red). PIL and display libraries expect **RGB** (Red-Green-Blue). Without conversion, all images appear with swapped red and blue channels.

Fix — one line:
```python
annotated_rgb = annotated_array[:, :, ::-1]
```
`[:, :, ::-1]` reverses the channel dimension — flipping BGR to RGB instantly.

---

### Detection Metadata in Response Headers
The `/detect/annotated` endpoint returns an image (not JSON). Detection statistics are passed back in HTTP headers so clients get both image and metadata in a single request:

```
X-Detection-Count : 3
X-Class-Summary   : {"person": 2, "laptop": 1}
```

---

### Confidence Threshold

Every detection has a confidence score (0.0 to 1.0). The threshold filters uncertain detections:

| Threshold | Effect |
|-----------|--------|
| 0.10 | Everything — including uncertain detections |
| 0.25 | Good default — catches most real objects |
| 0.50 | Only confident detections |
| 0.75 | Only very obvious, clearly visible objects |

The right value depends on the use case — security systems use low thresholds, automated labelling systems use high ones.

---

### Sandboxed Loading
YOLOv8 weights are loaded using Ultralytics' trusted pipeline — not raw `torch.load()`. This avoids the PyTorch 2.6 `weights_only` security change that caused a `_pickle.UnpicklingError` in earlier versions. Fixing it was a simple upgrade:
```bash
pip install --upgrade ultralytics
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root — confirms API is running |
| GET | `/health` | Health check for monitoring |
| GET | `/classes` | All 80 detectable class names |
| POST | `/detect` | Upload image → JSON detections |
| POST | `/detect/annotated` | Upload image → annotated JPEG |
| GET | `/docs` | Interactive Swagger UI |

---

### Example — JSON Detection

```python
import requests

with open('photo.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'file': f},
        params={'confidence': 0.25}
    )

result = response.json()
print(f"Found {result['detection_count']} objects")
for obj in result['detections']:
    print(f"  {obj['class']}: {obj['confidence']*100:.1f}%")
    print(f"  BBox: {obj['bbox']}")
```

Example response:
```json
{
    "detection_count": 2,
    "detections": [
        {
            "class": "person",
            "confidence": 0.9234,
            "bbox": {"x1": 100, "y1": 50, "x2": 300, "y2": 400}
        },
        {
            "class": "laptop",
            "confidence": 0.8712,
            "bbox": {"x1": 400, "y1": 150, "x2": 600, "y2": 350}
        }
    ],
    "class_summary": {"person": 1, "laptop": 1},
    "image_size": {"width": 1200, "height": 800}
}
```

### Example — cURL
```bash
curl -X POST "http://localhost:8000/detect" \
     -F "file=@photo.jpg" \
     -F "confidence=0.25"
```

---

## 📋 Detectable Classes

All 80 COCO object classes:

**People:** person

**Vehicles:** bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Traffic:** traffic light, fire hydrant, stop sign, parking meter, bench

**Animals:** bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Accessories:** backpack, umbrella, handbag, tie, suitcase

**Sports:** frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket

**Kitchen:** bottle, wine glass, cup, fork, knife, spoon, bowl

**Food:** banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

**Furniture:** chair, couch, potted plant, bed, dining table, toilet

**Electronics:** TV, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator

**Miscellaneous:** book, clock, vase, scissors, teddy bear, hair drier, toothbrush

---

## 🐛 Real Debugging Journey

After installing and running the project, it threw this error on first image upload:

```
_pickle.UnpicklingError: Weights only load failed.
In PyTorch 2.6, we changed the default value of the weights_only 
argument in torch.load from False to True.
```

**Root cause:** PyTorch 2.6 changed its default security settings for loading model weights. The older version of Ultralytics installed wasn't compatible with this change.

**Diagnosis:** Added `print(traceback.format_exc())` to the exception handler in FastAPI, which printed the full stack trace to the terminal — showing exactly which line caused the error and why.

**Fix:** One command:
```bash
pip install --upgrade ultralytics
```

The newer Ultralytics version handles PyTorch 2.6's new loading behaviour correctly.

**Lesson:** ML library version compatibility breaks regularly between releases. Reading error messages carefully and knowing which package to upgrade is a real production engineering skill.

---

## 🎬 Demo

https://github.com/user-attachments/assets/286646ac-8e28-4ccf-bbd1-e9f4add4f60e

---

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.11

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/hasanjaved180598/YOLOv8_Object_Detection.git
cd YOLOv8_Object_Detection

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# 3. Install dependencies
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Note:** YOLOv8 model weights (~6MB) download automatically on first run. An internet connection is required for the first launch only.

---

## 🚀 How to Run

Two terminals required — one for the API, one for the dashboard.

```bash
# Terminal 1 — Start FastAPI server
.venv\Scripts\python.exe api/main.py
# API running at: http://localhost:8000
# Interactive docs at: http://localhost:8000/docs

# Terminal 2 — Start Streamlit dashboard
.venv\Scripts\python.exe -m streamlit run app/app.py
# Dashboard at: http://localhost:8501
```

**Best images to test with:**
- Street scenes (people, cars, buses)
- Office photos (laptops, chairs, people)
- Kitchen photos (bottles, cups, food items)
- Outdoor scenes (animals, vehicles, people)

---

## 🔄 Project Workflow

```
Image Upload (Streamlit or direct API call)
    │
    ▼
api/main.py (FastAPI)
  ├── Validate file is an image (content_type check)
  ├── Read bytes → bytes_to_image() → PIL Image (RGB)
  └── Call run_detection() from src/model.py
    │
    ▼
src/model.py (YOLOv8)
  ├── Load yolov8n.pt (lazy — once on first call)
  ├── Single forward pass through 50-layer network
  ├── Extract: boxes (xyxy), confidences, class IDs
  ├── Filter detections below confidence threshold
  ├── result.plot() → draw coloured boxes + labels on image
  ├── Convert BGR → RGB ([:, :, ::-1])
  └── Return annotated PIL Image + detections list + class summary
    │
    ▼
/detect           → JSONResponse {detections, class_summary, image_size}
/detect/annotated → Response (JPEG bytes) + metadata in headers
    │
    ▼
app/app.py (Streamlit)
  ├── Display original vs annotated image side by side
  ├── Show metrics: object count, unique classes, confidence level
  ├── Bar chart of detected classes (Plotly)
  └── Expandable table: class, confidence, bbox coords per detection
```

---

## 📄 License

This project is licensed under the MIT License.

---

*The world is full of objects. Now your API knows them all. 🎯🌍*
