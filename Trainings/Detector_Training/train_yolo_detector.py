import os
import yaml
from ultralytics import YOLO
from Trainings.Detector_Training.preprocess_lables import PolygonPreprocessor


class YOLOTrainer:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def preprocess(self):
        print("Starting preprocessing...")
        train_proc = PolygonPreprocessor(
            image_dir=self.cfg['raw_train_images'],
            label_dir=self.cfg['raw_train_labels'],
            output_image_dir=self.cfg['processed_train_images'],
            output_label_dir=self.cfg['processed_train_labels']
        )
        train_proc.process()

        val_proc = PolygonPreprocessor(
            image_dir=self.cfg['raw_val_images'],
            label_dir=self.cfg['raw_val_labels'],
            output_image_dir=self.cfg['processed_val_images'],
            output_label_dir=self.cfg['processed_val_labels']
        )
        val_proc.process()

    def create_data_yaml(self):
        print("Creating data.yaml...")
        os.makedirs(os.path.dirname(self.cfg['yaml_output_path']), exist_ok=True)
        # Absolute root directory (e.g., your project root)
        root_dir = self.cfg.get("root_dir", os.getcwd())  # fallback to CWD if not defined

        train_dir = os.path.join(root_dir, os.path.dirname(self.cfg['processed_train_images']))
        val_dir = os.path.join(root_dir, os.path.dirname(self.cfg['processed_val_images']))

        with open(self.cfg['yaml_output_path'], "w") as f:
            f.write(f"""
train: {train_dir}
val: {val_dir}

nc: 1
names: ['text']
""".strip())

        print(f"data.yaml created at {self.cfg['yaml_output_path']}")

    def train(self):
        print("Starting YOLO training...")
        model = YOLO(self.cfg['model_name'])
        results = model.train(
            data=self.cfg['yaml_output_path'],
            epochs=self.cfg['epochs'],
            imgsz=self.cfg['img_size']
        )
        print("Training complete.")
        return results

    def run(self):
        self.preprocess()
        self.create_data_yaml()
        return self.train()


if __name__ == "__main__":
    trainer = YOLOTrainer("Configs/config.yaml")
    trainer.run()
