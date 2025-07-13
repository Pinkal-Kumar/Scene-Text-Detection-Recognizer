import os
import re
import shutil
from PIL import Image


class PolygonPreprocessor:
    def __init__(self, image_dir, label_dir, output_image_dir, output_label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_image_dir = output_image_dir
        self.output_label_dir = output_label_dir
        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_label_dir, exist_ok=True)

    @staticmethod
    def normalize_polygon_coords(coords, image_width, image_height):
        return [0] + [
            round(coords[i] / image_width, 6) if i % 2 == 0
            else round(coords[i] / image_height, 6)
            for i in range(len(coords))
        ]

    @staticmethod
    def extract_coords_and_transcription(parts):
        x_line = [s for s in parts if s.strip().startswith("x:")][0]
        y_line = [s for s in parts if s.strip().startswith("y:")][0]
        trans_line = [s for s in parts if 'transcriptions' in s][0]

        x_vals = list(map(int, re.findall(r'\d+', x_line)))
        y_vals = list(map(int, re.findall(r'\d+', y_line)))
        transcription = re.findall(r"u?'(.*?)'", trans_line)[0]
        coords = [val for pair in zip(x_vals, y_vals) for val in pair]
        return coords, transcription

    def process(self):
        for label_file in os.listdir(self.label_dir):
            if not label_file.endswith(".txt"):
                continue

            label_path = os.path.join(self.label_dir, label_file)
            image_name = label_file.split("_")[-1].replace(".txt", ".jpg")
            image_path = os.path.join(self.image_dir, image_name)

            if not os.path.exists(image_path):
                print(f"Image missing for {label_file}")
                continue

            with Image.open(image_path) as img:
                w, h = img.size

            shutil.copy(image_path, os.path.join(self.output_image_dir, image_name))

            yolo_labels = []
            with open(label_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    parts = line.strip().split(",")
                    try:
                        coords, _ = self.extract_coords_and_transcription(parts)
                        if len(coords) < 8:
                            continue
                        yolo_coords = self.normalize_polygon_coords(coords, w, h)
                        yolo_labels.append(" ".join(map(str, yolo_coords)))
                    except Exception as e:
                        print(f"Error parsing line {i+1} in {label_file}: {e}")
                        continue

            out_label_path = os.path.join(self.output_label_dir, label_file.split("_")[-1])
            with open(out_label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_labels))

        print(f"Finished processing: {self.label_dir}")
