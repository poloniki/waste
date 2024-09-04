import cv2
import requests


from image_prediction import create_image


import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def video_frame_callback(frame):

    format = "bgr24"
    img = frame.to_ndarray(format=format)

    _, encoded_image = cv2.imencode(".jpg", img)
    bytes_data = encoded_image.tobytes()

    res = requests.post(
        url="http://34.118.100.170:8000/upload_image",
        # url=" http://127.0.0.1:8000/upload_image",
        # url="http://65.108.32.135:8000/upload_image",
        files={"img": bytes_data},
    ).json()["boundsboxes"]

    created_image = create_image(img, res)  # Assuming create_image is defined somewhere

    return av.VideoFrame.from_ndarray(created_image, format=format)


webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    mode=WebRtcMode.SENDONLY,
)
