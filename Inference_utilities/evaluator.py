import os
import cv2
import yaml
import torch
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

from Inference_utilities.precision_tracker import PrecisionTracker
from Inference_utilities.ground_truth_utils import load_ground_truth

class SceneTextEvaluator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.yolo_model = YOLO(cfg["yolo_weights"])
        self.processor = TrOCRProcessor.from_pretrained(cfg["recognizer_model"])
        self.trocr = VisionEncoderDecoderModel.from_pretrained(cfg["recognizer_model"]).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.trocr.to(self.device)

        self.image_dir = cfg["image_dir"]
        self.label_dir = cfg["label_dir"]
        self.max_images = cfg.get("max_images", 100)
        self.tracker = PrecisionTracker(method="exact", threshold=cfg.get("text_thresh", 0.8) * 100)

    def crop_polygon(self, image, polygon):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        x, y, w, h = cv2.boundingRect(points)
        return cv2.bitwise_and(image, image, mask=mask)[y:y+h, x:x+w]

    def recognize_text(self, image_pil):
        pixel_values = self.processor(images=image_pil, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            ids = self.trocr.generate(pixel_values)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def extract_texts(self, predictions):
        return [p['text'] for p in predictions if 'text' in p]

    def evaluate(self):
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))])
        count = 0

        for img_file in image_files:
            if count >= self.max_images:
                break

            image_path = os.path.join(self.image_dir, img_file)
            label_path = os.path.join(self.label_dir, f"poly_gt_{os.path.splitext(img_file)[0]}.txt")
            if not os.path.exists(label_path):
                continue

            image = cv2.imread(image_path)
            gts = load_ground_truth(label_path)

            results = self.yolo_model(image)[0]
            predictions = []

            if results.masks:
                polys_boxes = sorted(zip(results.masks.xy, results.boxes.xyxy.tolist()), key=lambda b: b[1][1])
                for seg, box in polys_boxes:
                    polygon = [(float(x), float(y)) for x, y in seg]
                    cropped = self.crop_polygon(image, polygon)
                    if cropped.size == 0:
                        continue
                    pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    pred_text = self.recognize_text(pil_img)
                    predictions.append({'polygon': polygon, 'text': pred_text})

                gts_text = self.extract_texts(gts)
                preds_text = self.extract_texts(predictions)
                print("-------gts text list-------> :",gts_text)
                print("-------preds text list-----> :",preds_text)
                if len(gts_text)>0:
                    self.tracker.update(gts_text, preds_text)
                count += 1
                print(f"Processed {count} image(s): {img_file}")

        precision = self.tracker.get_precision()
        print(f"Precision over {count} image(s): {precision:.4f}")
        return precision
