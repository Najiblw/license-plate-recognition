from __future__ import annotations

from pathlib import Path

import pytesseract

from .config import AppConfig, build_tesseract_config


def configure_tesseract(config: AppConfig) -> None:
    if Path(config.tesseract_cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd


def extract_text_from_plate(plate_image, config: AppConfig) -> str:
    configure_tesseract(config)
    tesseract_config = build_tesseract_config(config)
    try:
        return pytesseract.image_to_string(plate_image, config=tesseract_config)
    except pytesseract.TesseractNotFoundError:
        return ""
