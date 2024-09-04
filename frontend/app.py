import cv2
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av

# Configuration for the WebRTC peer connection (STUN server for NAT traversal)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Initialize a frame counter outside the function
frame_counter = 0


# Callback for processing video frames
def video_frame_callback(frame):
    global frame_counter  # Access the global frame counter
    format = "bgr24"
    img = frame.to_ndarray(format=format)

    # Increment the frame counter
    frame_counter += 1

    # Skip the first 4 frames, only process every 5th frame
    if frame_counter % 5 != 0:
        return av.VideoFrame.from_ndarray(img, format=format)

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
    return av.VideoFrame.from_ndarray(created_image, format=format)


# WebRTC streamer in SENDONLY mode
webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,  # Callback for video frame processing
    rtc_configuration=RTC_CONFIGURATION,  # WebRTC configuration
)
