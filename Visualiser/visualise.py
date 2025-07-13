import os
import cv2
import argparse
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
import torch
import time

def main(args):
    # Load YOLOv8-seg model
    print(f"Loading detector weights from {args.detector_weights}...")
    model = YOLO(args.detector_weights)

    # Load TrOCR
    print(f"Loading recognizer weights from {args.recognizer_weights}...")
    processor = TrOCRProcessor.from_pretrained(args.recognizer_weights)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(args.recognizer_weights)

    # Load image
    print(f"Reading input image from {args.input}...")
    original_img = cv2.imread(args.input)
    if original_img is None:
        raise FileNotFoundError(f"Input image not found at {args.input}")

    # Run detection
    print("Running YOLOv8 segmentation...")
    results = model(args.input)[0]

    # Process each detected polygon
    for i, polygon in enumerate(results.masks.xy):
        polygon = cv2.convexHull(polygon.astype('int32'))
        x, y, w, h = cv2.boundingRect(polygon)
        cropped = original_img[y:y+h, x:x+w]

        # Convert to PIL and run through TrOCR
        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values

        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"[{i+1}] Recognized Text: {transcription}")

        # Annotate image
        cv2.polylines(original_img, [polygon], isClosed=True, color=(0,255,0), thickness=2)
        cv2.putText(original_img, transcription, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save result
    vis_output_dir = args.vis_output
    os.makedirs(vis_output_dir, exist_ok=True)
    filename = f"output_{int(time.time())}.jpg"
    output_path = os.path.join(vis_output_dir, filename)
    cv2.imwrite(output_path, original_img)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scene Text Detection and Recognition")
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--detector_weights', type=str, required=True, help='Path to YOLOv8-seg detector weights')
    parser.add_argument('--recognizer_weights', type=str, default="microsoft/trocr-base-stage1", help='HuggingFace model ID or path for TrOCR recognizer')
    parser.add_argument('--vis_output', type=str, required=True, help='Path to save output visualization image')

    args = parser.parse_args()
    main(args)
