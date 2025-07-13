import os
import sys
import uuid
import cv2
import yaml
import torch
import logging
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

# ---------------- FastAPI Setup ----------------
app = FastAPI(title="Scene Text Detection & Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Inference Engine (Global) ----------------
inference_engine = None
config = None

# ---------------- Startup Hook ----------------
@app.on_event("startup")
def load_inference_engine():
    global inference_engine, config

    config_path = os.path.join("Configs", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    inference_engine = SceneTextEvaluator(config_path=config_path)

    # Load models using config
    inference_engine.text_thresh = config.get("text_thresh", 0.8)
    inference_engine.max_images = config.get("max_images", 1)

    logging.info("Inference engine and models loaded successfully.")

# ---------------- Helper Function ----------------
def read_imagefile(file) -> np.ndarray:
    image = Image.open(file.file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# ---------------- Inference Endpoint ----------------
@app.post("/infer/")
async def infer_text_from_image(image: UploadFile = File(...)):
    try:
        global inference_engine
        if inference_engine is None:
            raise RuntimeError("Inference engine not initialized.")

        # Save and read input image
        image_np = read_imagefile(image)
        tmp_name = f"temp_{uuid.uuid4().hex[:6]}.jpg"
        cv2.imwrite(tmp_name, image_np)

        # Inference
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
        return JSONResponse(status_code=200, content={"texts": final_text})

    except Exception as e:
        logging.exception("Inference failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapiapp:app", host="0.0.0.0", port=8000, reload=True)
