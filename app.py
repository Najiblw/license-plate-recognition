from __future__ import annotations

import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.config import AppConfig
from src.main import process_image_array


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL image from Streamlit into the BGR format used by OpenCV."""

    rgb_image = image.convert("RGB")
    return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV output back to RGB so Streamlit shows the right colors."""

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_config() -> AppConfig:
    config = AppConfig()
    # Community Cloud installs Tesseract through packages.txt. On Linux the
    # command is available as "tesseract" on PATH.
    config.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")
    return config


def main() -> None:
    st.set_page_config(page_title="License Plate Recognition", layout="wide")

    st.title("License Plate Recognition")
    st.write(
        "Upload a vehicle image to detect a license plate and extract its text "
        "using OpenCV and Tesseract OCR."
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is None:
        st.info("Upload a vehicle image to run the pipeline.")
        return

    uploaded_image = Image.open(uploaded_file)
    image_bgr = pil_to_bgr(uploaded_image)
    config = build_config()

    with st.spinner("Running plate detection and OCR..."):
        result = process_image_array(image_bgr, config)

    preview_col, result_col = st.columns(2)

    with preview_col:
        st.subheader("Uploaded Image")
        st.image(uploaded_image, use_container_width=True)

    with result_col:
        st.subheader("Detection Result")
        if result.annotated_image is not None:
            st.image(
                bgr_to_rgb(result.annotated_image),
                use_container_width=True,
            )
        else:
            st.warning("No plate region was detected in the uploaded image.")

    st.subheader("Extracted License Plate Text")
    if result.detected_text:
        st.success(result.detected_text)
    else:
        st.error("No license plate text was extracted.")
    st.caption(result.message)

    detail_col_1, detail_col_2 = st.columns(2)

    with detail_col_1:
        st.subheader("Detected Plate Region")
        if result.plate_image is not None:
            st.image(
                bgr_to_rgb(result.plate_image),
                use_container_width=True,
            )
        else:
            st.info("Plate crop not available.")

    with detail_col_2:
        st.subheader("OCR Input")
        if result.ocr_image is not None:
            st.image(
                result.ocr_image,
                clamp=True,
                use_container_width=True,
            )
        else:
            st.info("OCR-ready image not available.")


if __name__ == "__main__":
    main()
