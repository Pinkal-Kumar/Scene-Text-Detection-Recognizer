import os
import sys
import uuid
import cv2
import yaml
import torch
import logging
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from Inference_utilities.evaluator import SceneTextEvaluator

# Add current directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ---------------- Logger Setup ----------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "inference.log")

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- Flask App Setup ----------------
app = Flask(__name__)
CORS(app)

# ---------------- Inference Engine (Global) ----------------
inference_engine = None
config = None

# ---------------- Startup Hook ----------------
@app.before_request
def load_inference_engine():
    global inference_engine, config

    config_path = os.path.join("Configs", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    inference_engine = SceneTextEvaluator(config_path=config_path)
    inference_engine.text_thresh = config.get("text_thresh", 0.8)
    inference_engine.max_images = config.get("max_images", 1)

    logging.info("Inference engine and models loaded successfully.")

# ---------------- Helper Function ----------------
def read_imagefile(file) -> np.ndarray:
    image = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# ---------------- Inference Endpoint ----------------
@app.route("/infer/", methods=["POST"])
def infer_text_from_image():
    try:
        global inference_engine
        if inference_engine is None:
            raise RuntimeError("Inference engine not initialized.")

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        image_np = read_imagefile(image_file)
        tmp_name = f"temp_{uuid.uuid4().hex[:6]}.jpg"
        cv2.imwrite(tmp_name, image_np)

        results = inference_engine.yolo_model(tmp_name)[0]
        predictions = []

        if results.masks:
            polys_boxes = sorted(zip(results.masks.xy, results.boxes.xyxy.tolist()), key=lambda b: b[1][1])
            for seg, _ in polys_boxes:
                polygon = [(float(x), float(y)) for x, y in seg]
                cropped = inference_engine.crop_polygon(image_np, polygon)
                if cropped.size == 0:
                    continue
                pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                text = inference_engine.recognize_text(pil_img)
                predictions.append(text)

        os.remove(tmp_name)
        final_text = " ".join(predictions)
        log_text = f"Recognised text from the given image : {final_text}"
        logging.info(log_text)
        return jsonify({"texts": final_text})

    except Exception as e:
        logging.exception("Inference failed")
        return jsonify({"error": str(e)}), 500

# ---------------- Run Server ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
