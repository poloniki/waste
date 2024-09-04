import cv2
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from image_prediction import (
    create_image,
)  # Assuming this is defined in your environment

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# Define a class for the video processing
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the frame to an ndarray (OpenCV format)
        img = frame.to_ndarray(format="bgr24")

        # Encode the image to bytes to send via POST request
        _, encoded_image = cv2.imencode(".jpg", img)
        bytes_data = encoded_image.tobytes()

        # Send the image to the external server for processing
        res = requests.post(
            url="http://34.118.100.170:8000/upload_image",
            files={"img": bytes_data},
        ).json()["boundsboxes"]

        # Create the resulting image with the bounding boxes (from your custom function)
        created_image = create_image(img, res)  # Assuming create_image is defined

        # Return the new frame to be displayed
        return av.VideoFrame.from_ndarray(created_image, format="bgr24")


# Use the updated API with video_processor_factory
webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,  # Use video_processor_factory
    rtc_configuration=RTC_CONFIGURATION,
)
