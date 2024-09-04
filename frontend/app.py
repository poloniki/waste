import cv2
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
from image_prediction import create_image
import av

# Configuration for the WebRTC peer connection (STUN server for NAT traversal)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# Define the VideoProcessor class to handle frame processing
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_counter = 0  # Initialize frame counter

    def recv(self, frame):
        self.frame_counter += 1  # Increment the frame counter

        # Convert the incoming frame to a numpy array (BGR format)
        img = frame.to_ndarray(format="bgr24")

        # Skip the first 4 frames, only process every 5th frame
        if self.frame_counter % 5 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Process the 5th frame (send it to the server for prediction)
        _, encoded_image = cv2.imencode(".jpg", img)
        bytes_data = encoded_image.tobytes()

        # Send the image to the server to get bounding boxes
        res = requests.post(
            url="http://34.118.100.170:8000/upload_image",
            files={"img": bytes_data},  # Send as form-data
        ).json()["boundsboxes"]

        # Assuming you have a function to draw bounding boxes on the image
        created_image = create_image(img, res)

        # Return the modified frame with bounding boxes
        return av.VideoFrame.from_ndarray(created_image, format="bgr24")


# Create the WebRTC streamer using the VideoProcessor class
webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,  # Use the new class for video processing
    rtc_configuration=RTC_CONFIGURATION,  # WebRTC configuration
)
