import streamlit as st
import cv2
import numpy as np

st.title("Virtual Mirror with Fun Filters")

# Load your overlay image (transparent PNG)
# Example: hat.png with transparency
hat = cv2.imread('hat.png', cv2.IMREAD_UNCHANGED)

# Camera input
uploaded_file = st.camera_input("Look into your webcam!")

if uploaded_file:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------------------------------
    # Apply overlay (hat) on face
    # -------------------------------
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Resize hat to face width
        hat_resized = cv2.resize(hat, (w, int(h/2)))

        # Position: top of the face
        y1, y2 = y - hat_resized.shape[0], y
        x1, x2 = x, x + hat_resized.shape[1]

        if y1 < 0:  # boundary check
            y1 = 0

        # Overlay transparent hat
        alpha_hat = hat_resized[:, :, 3] / 255.0
        for c in range(3):
            frame_rgb[y1:y2, x1:x2, c] = alpha_hat * hat_resized[:, :, c] + (1 - alpha_hat) * frame_rgb[y1:y2, x1:x2, c]

    # Display final image
    st.image(frame_rgb)
