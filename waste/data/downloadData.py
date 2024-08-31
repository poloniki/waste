import logging

from roboflow import Roboflow

from waste.params import *


def load_data():
    rf = Roboflow(api_key="NtHDWWC0JhDMlxNNhpDb")
    project = rf.workspace("tom-rowland-03ams").project("taco-wikc7")
    version = project.version(3)
    dataset = version.download("yolov8-obb", location=LOCAL_DATA_PATH)
