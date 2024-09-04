from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
from waste.ml_logic.model import load_best_weights
import logging
from waste.params import *

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
app.state.model = YOLO(load_best_weights())


@app.get("/")
def hello_world():
    return {"hello": "world"}


@app.post("/upload_image")
async def receive_image(img: UploadFile = File(...)):
    model = app.state.model
    contents = await img.read()

    nparr = np.frombuffer(contents, np.uint8)

    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    bound_boxes = []
    # We need to adjust stride
    prediction = model(
        cv2_img,
        device="cuda:0",
        imgsz=IMG_SIZE,
        conf=0.1,
        vid_stride=10,
        # augment=True,
        stream_buffer=True,
        # agnostic_nms=True,
        # retina_masks=True,
    )

    for box in prediction[0].boxes:
        card = box.cls.tolist()[0]
        card_name = prediction[0].names[card]
        confidence = round(box.conf[0].item(), 2)

        dict = {
            "Object type": card_name,
            "Coordinates": box.xyxy.tolist()[0],  # This line is changed
            "Probability": confidence,
        }
        bound_boxes.append(dict)

    return {"boundsboxes": bound_boxes}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    model = app.state.model

    try:
        while True:
            data = await websocket.receive_bytes()

            # Decode the frame
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Predict
            prediction = model(frame, device="cuda:0", imgsz=IMG_SIZE, conf=0.1)

            bound_boxes = []
            for box in prediction[0].boxes:
                card = box.cls.tolist()[0]
                card_name = prediction[0].names[card]
                confidence = round(box.conf[0].item(), 2)
                bound_boxes.append(
                    {
                        "Object type": card_name,
                        "Coordinates": box.xyxy.tolist()[0],
                        "Probability": confidence,
                    }
                )

            await websocket.send_json({"boundsboxes": bound_boxes})

    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()
