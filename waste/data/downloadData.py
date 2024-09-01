import logging

from roboflow import Roboflow

from waste.params import *


def load_data():
    rf = Roboflow(api_key="NtHDWWC0JhDMlxNNhpDb")
    project = rf.workspace("tom-rowland-03ams").project("taco-wikc7")
    version = project.version(3)
    dataset = version.download("yolov8-obb", location=LOCAL_DATA_PATH)

    # rf = Roboflow(api_key="NtHDWWC0JhDMlxNNhpDb")
    # project = rf.workspace("taco-t7kkz").project("taco-dataset-ql1ng")
    # version = project.version(15)
    # dataset = version.download("yolov9", location=LOCAL_DATA_PATH)


if __name__ == "__main__":
    load_data()
