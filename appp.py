import streamlit as st
import cv2
import numpy as np

st.title("Virtual Mirror Demo")

uploaded_file = st.camera_input("Look into your webcam!")

if uploaded_file:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for displaying
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    st.image(frame_rgb)
