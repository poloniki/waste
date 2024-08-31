import logging
import os

import comet_ml
from ultralytics import YOLO

from waste.ml_logic.model import load_best_weights, update_production_model
from waste.params import *


def train_model():
    # Initialize Comet ML API connection
    comet_ml.init()

    try:
        weights = load_best_weights()
    except Exception as error:
        logging.warning(f"❌ Could not load weights: {error}")
        weights = os.path.join(WEIGHTS_PATH, "yolov8n.pt")

    model = YOLO(weights)

    model.train(
        data=os.path.join(LOCAL_DATA_PATH, "data.yaml"),
        epochs=NUM_EPOCHS,
        imgsz=IMG_SIZE,
        patience=20,
        save_dir=RUNS_PATH,
        plots=True,
    )

    update_production_model()


# Main execution
if __name__ == "__main__":
    train_model()
