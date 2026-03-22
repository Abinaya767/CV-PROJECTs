from ipywidgets import FileUpload, VBox, Output, Dropdown, Button
from IPython.display import display, clear_output, FileLink
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# -----------------------------
# Widgets
# -----------------------------
upload = FileUpload(accept='image/*', multiple=False)

filter_dropdown = Dropdown(
    options=[
        "Auto Scan (Best)",
        "Black & White",
        "High Contrast",
        "Color",
        "Pencil",
        "Warm Tone 🔥",
        "Cool Tone ❄️",
        "Vintage 🟤",
        "Bright 🌞",
        "HD Enhance ✨",
        "Full HD (1080p) 🖥️",
        "4K Ultra HD 📺"
    ],
    value="Auto Scan (Best)",
    description="Mode:"
)

download_btn = Button(description="Download PDF", button_style='success')

out = Output()

display(VBox([upload, filter_dropdown, download_btn, out]))

processed_image = None

# -----------------------------
# SAFE SCAN
# -----------------------------
def safe_scan(gray):
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    scan = cv2.adaptiveThreshold(
        norm, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15, 5
    )

    if np.mean(scan) > 240:
        _, scan = cv2.threshold(norm, 130, 255, cv2.THRESH_BINARY)

    return scan

# -----------------------------
# PROCESS IMAGE
# -----------------------------
def process_image(change):
    global processed_image

    with out:
        clear_output()

        if not upload.value:
            print("Upload image first!")
            return

        file = upload.value[0]
        img_bytes = file['content']

        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        img = cv2.resize(img, (800, 600))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # -----------------------------
        # FILTERS
        # -----------------------------
        if filter_dropdown.value == "Auto Scan (Best)":
            processed_image = safe_scan(gray)

        elif filter_dropdown.value == "Black & White":
            _, processed_image = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

        elif filter_dropdown.value == "High Contrast":
            processed_image = cv2.convertScaleAbs(gray, alpha=1.8, beta=20)

        elif filter_dropdown.value == "Color":
            processed_image = img

        elif filter_dropdown.value == "Pencil":
            inv = cv2.bitwise_not(gray)
            blur = cv2.GaussianBlur(inv, (21,21), 0)
            inv_blur = cv2.bitwise_not(blur)
            processed_image = cv2.divide(gray, inv_blur, scale=256.0)

        elif filter_dropdown.value == "Warm Tone 🔥":
            processed_image = cv2.add(img, np.array([20,10,0], dtype=np.uint8))

        elif filter_dropdown.value == "Cool Tone ❄️":
            processed_image = cv2.add(img, np.array([0,10,20], dtype=np.uint8))

        elif filter_dropdown.value == "Vintage 🟤":
            kernel = np.array([[0.272,0.534,0.131],
                               [0.349,0.686,0.168],
                               [0.393,0.769,0.189]])
            processed_image = cv2.transform(img, kernel)

        elif filter_dropdown.value == "Bright 🌞":
            processed_image = cv2.convertScaleAbs(img, alpha=1.3, beta=40)

        # -----------------------------
        # HD / 4K ENHANCEMENT
        # -----------------------------
        elif filter_dropdown.value == "HD Enhance ✨":
            processed_image = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

        elif filter_dropdown.value == "Full HD (1080p) 🖥️":
            processed_image = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)

        elif filter_dropdown.value == "4K Ultra HD 📺":
            processed_image = cv2.resize(img, (3840, 2160), interpolation=cv2.INTER_CUBIC)

        # -----------------------------
        # ADD BORDER
        # -----------------------------
        processed_image = cv2.copyMakeBorder(
            processed_image, 20, 20, 20, 20,
            cv2.BORDER_CONSTANT, value=[255,255,255]
        )

        # -----------------------------
        # DISPLAY
        # -----------------------------
        plt.figure(figsize=(8,5))

        if len(processed_image.shape) == 2:
            plt.imshow(processed_image, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        plt.title("Enhanced Output")
        plt.axis("off")
        plt.show()

# -----------------------------
# DOWNLOAD PDF
# -----------------------------
def download_pdf(b):
    global processed_image

    with out:
        if processed_image is None:
            print("❌ No image to download!")
            return

        try:
            filename = "scanned_output.pdf"

            if len(processed_image.shape) == 2:
                pil_img = Image.fromarray(processed_image)
            else:
                pil_img = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

            pil_img = pil_img.convert("RGB")
            pil_img.save(filename, "PDF")

            print("✅ PDF Ready!")
            print("👇 Click below to download:")
            display(FileLink(filename))

        except Exception as e:
            print("❌ Error:", e)

# -----------------------------
# EVENTS
# -----------------------------
upload.observe(process_image, names='value')
filter_dropdown.observe(process_image, names='value')
download_btn.on_click(download_pdf)
