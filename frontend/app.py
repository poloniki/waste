import cv2
import requests

# import streamlit as st

from image_prediction import create_image

# from streamlit_webrtc import RTCConfiguration, webrtc_streamer

# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# )


# class VideoProcessor:
#     def __init__(self):
#         self.count = 0

#     def recv(self, frame):
#         print(1)
#         format = "bgr24"
#         img = frame.to_ndarray(format=format)

#         _, encoded_image = cv2.imencode(".jpg", img)
#         bytes_data = encoded_image.tobytes()

#         res = requests.post(
#             url="http://localhost:8000/upload_image",
#             files={"img": bytes_data},
#         ).json()["boundsboxes"]

#         created_image = create_image(
#             img, res
#         )  # Assuming create_image is defined somewhere

#         return av.VideoFrame.from_ndarray(created_image, format=format)


# def main():
#     webrtc_ctx = webrtc_streamer(
#         key="WYH",
#         # mode=WebRtcMode.SENDRECV,
#         rtc_configuration=RTC_CONFIGURATION,
#         video_processor_factory=VideoProcessor,
#         media_stream_constraints={"video": True, "audio": False},
#         # async_processing=False,
#     )


# if __name__ == "__main__":
#     main()

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av


def video_frame_callback(frame):

    format = "bgr24"
    img = frame.to_ndarray(format=format)

    _, encoded_image = cv2.imencode(".jpg", img)
    bytes_data = encoded_image.tobytes()

    res = requests.post(
        # url="http://34.116.237.23:8000/upload_image",
        # url=" http://127.0.0.1:8080/upload_image",
        files={"img": bytes_data},
    ).json()["boundsboxes"]

    created_image = create_image(img, res)  # Assuming create_image is defined somewhere

    return av.VideoFrame.from_ndarray(created_image, format=format)


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
