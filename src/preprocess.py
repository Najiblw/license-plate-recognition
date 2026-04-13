from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import AppConfig


def enhance_contrast(gray: np.ndarray, config: AppConfig) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_grid_size,
    )
    return clahe.apply(gray)


def resize_image(image: np.ndarray, width: int) -> np.ndarray:
    height, current_width = image.shape[:2]
    if current_width <= width:
        return image.copy()

    scale = width / float(current_width)
    resized_height = int(height * scale)
    return cv2.resize(image, (width, resized_height), interpolation=cv2.INTER_AREA)


def preprocess_for_plate_detection(
    image: np.ndarray,
    config: AppConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    resized = resize_image(image, config.resize_width)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_contrast(gray, config)
    filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)
    edged = cv2.Canny(filtered, config.canny_threshold_1, config.canny_threshold_2)
    return resized, enhanced, edged


def preprocess_plate_for_ocr(
    plate_image: np.ndarray,
    config: AppConfig,
) -> np.ndarray:
    start_row = int(plate_image.shape[0] * config.ocr_lower_region_start)
    focused_region = plate_image[start_row:, :]
    if focused_region.size == 0:
        focused_region = plate_image

    gray = cv2.cvtColor(focused_region, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_contrast(gray, config)
    filtered = cv2.bilateralFilter(enhanced, 7, 25, 25)
    upscaled = cv2.resize(
        filtered,
        None,
        fx=config.ocr_scale_factor,
        fy=config.ocr_scale_factor,
        interpolation=cv2.INTER_CUBIC,
    )
    thresholded = cv2.adaptiveThreshold(
        upscaled,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        config.adaptive_block_size,
        config.adaptive_c,
    )
    kernel = np.ones(
        (config.morphology_kernel_size, config.morphology_kernel_size),
        dtype=np.uint8,
    )
    closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened
