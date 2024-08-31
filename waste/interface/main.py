import os
from waste.params import *
from waste.data.downloadData import load_data
from waste.ml_logic.train import train_model
import logging

logging.basicConfig(level=logging.INFO)


def main():
    yaml_path = os.path.join(LOCAL_DATA_PATH, "data.yaml")
    data_exists = os.path.isfile(yaml_path)

    if not data_exists:
        load_data()
    else:
        logging.info("✅ Dataset already exists")

    train_model()


if __name__ == "__main__":
    main()
