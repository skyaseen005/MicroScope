from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
import json

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(MODEL_PATH)
print(f"✅ Model loaded | Classes: {model.names}")

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Run inference
    results = model(image, conf=0.25)
    result = results[0]

    detections = []
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="#00FFCC", width=3)
            
            # Draw label background
            text = f"{label} {conf:.0%}"
            text_bbox = draw.textbbox((x1, y1 - 22), text)
            draw.rectangle(text_bbox, fill="#00FFCC")
            draw.text((x1, y1 - 22), text, fill="#000000")

            detections.append({
                "label": label,
                "confidence": round(conf, 4),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "width_px": round(x2 - x1),
                "height_px": round(y2 - y1),
            })

    # Encode annotated image to base64
    buf = io.BytesIO()
    annotated_image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    total = len(detections)
    avg_conf = round(sum(d["confidence"] for d in detections) / total, 4) if total else 0

    return jsonify({
        "total_detections": total,
        "average_confidence": avg_conf,
        "detections": detections,
        "annotated_image": f"data:image/png;base64,{encoded}",
        "image_size": {"width": image.width, "height": image.height},
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
