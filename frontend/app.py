import cv2
import requests
from image_prediction import create_image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av
import os
from twilio.rest import Client

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

    res = requests.post(
        url="http://34.118.100.170:8000/upload_image",
        files={"img": bytes_data},
    ).json()["boundsboxes"]

    created_image = create_image(img, res)  # Assuming create_image is defined somewhere

    return av.VideoFrame.from_ndarray(created_image, format=format)


webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
)
