import os

LOCAL_DATA_PATH = os.path.expanduser("~/.data/waste")
WEIGHTS_PATH = os.path.join(LOCAL_DATA_PATH, "weights")
RUNS_PATH = os.path.join(LOCAL_DATA_PATH, "cache")
NUM_EPOCHS = int(os.environ["NUM_EPOCHS"])
COMET_API_KEY = os.environ["COMET_API_KEY"]
COMET_PROJECT_NAME = os.environ["COMET_PROJECT_NAME"]
COMET_MODEL_NAME = os.environ["COMET_MODEL_NAME"]
COMET_WORKSPACE_NAME = os.environ["COMET_WORKSPACE_NAME"]
IMG_SIZE = int(os.environ["IMG_SIZE"])

os.makedirs(LOCAL_DATA_PATH, exist_ok=True)
os.makedirs(WEIGHTS_PATH, exist_ok=True)
os.makedirs(RUNS_PATH, exist_ok=True)
