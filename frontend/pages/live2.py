import cv2
import requests
from image_prediction import create_image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av
import os
from twilio.rest import Client
from ultralytics import YOLO


@st.cache_resource
def load_model():
    model = YOLO("frontend/pages/best-59319.pt")
    return model


model = load_model()

account_sid = st.secrets["twilio"]["account_sid"]
auth_token = st.secrets["twilio"]["auth_token"]
client = Client(account_sid, auth_token)

token = client.tokens.create()

RTC_CONFIGURATION = RTCConfiguration({"iceServers": token.ice_servers})


def video_frame_callback(frame):

    format = "bgr24"
    img = frame.to_ndarray(format=format)

    _, encoded_image = cv2.imencode(".jpg", img)
    bytes_data = encoded_image.tobytes()

    bound_boxes = []
    prediction = model(
        img,
        device="cuda:0",
        imgsz=640,
        conf=0.1,
        vid_stride=10,
        stream_buffer=True,
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

    created_image = create_image(
        img, bound_boxes
    )  # Assuming create_image is defined somewhere

    return av.VideoFrame.from_ndarray(created_image, format=format)


webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
)
