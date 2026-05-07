# MicroScope — Microplastic Detection App

AI-powered web application to detect microplastic particles in images using your trained YOLOv8 model.

## Project Structure

```
microplastic-app/
├── app.py               # Flask backend
├── best.pt              # Your trained YOLOv8 model ← PUT YOUR MODEL HERE
├── requirements.txt     # Python dependencies
└── templates/
    └── index.html       # Frontend UI
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your model
Copy your `best.pt` file into the `microplastic-app/` folder (same directory as `app.py`).

### 3. Run the server
```bash
python app.py
```

### 4. Open browser
Visit: [http://localhost:5000](http://localhost:5000)

---

## How it works

1. **Upload** a microscopy or water sample image (JPG, PNG, BMP, TIFF)
2. Click **Analyze** — the image is sent to the Flask backend
3. The YOLOv8 model runs inference and detects microplastic particles
4. Results are shown with:
   - Annotated image with bounding boxes
   - Particle count, average confidence, risk level
   - Per-particle breakdown with confidence bars

## API Endpoint

**POST** `/detect`  
- Body: `multipart/form-data` with field `image`
- Response JSON:
```json
{
  "total_detections": 5,
  "average_confidence": 0.82,
  "detections": [
    {
      "label": "Microplastic",
      "confidence": 0.91,
      "bbox": [x1, y1, x2, y2],
      "width_px": 34,
      "height_px": 28
    }
  ],
  "annotated_image": "data:image/png;base64,...",
  "image_size": { "width": 640, "height": 480 }
}
```

## Risk Levels
| Particles | Risk Level |
|-----------|-----------|
| 0         | Clean     |
| 1 – 3     | Low       |
| 4 – 8     | Medium    |
| 9+        | High      |
