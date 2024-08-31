import logging

from roboflow import Roboflow

from waste.params import *


def load_data():
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("augmented-startups").project("playing-cards-ow27d")
    project.version(2).download("yolov8", location=LOCAL_DATA_PATH)
    logging.info("✅ Succesufully downloaded dataset from the roboflow")
