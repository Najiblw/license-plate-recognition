from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import AppConfig
from .ocr import extract_text_from_plate
from .plate_detector import PlateDetectionResult, detect_plate
from .postprocess import clean_plate_text, is_plausible_plate
from .preprocess import preprocess_plate_for_ocr
from .utils import draw_plate_box, save_image


@dataclass(slots=True)
class PipelineResult:
    image_path: Path
    detected_text: str
    is_valid: bool
    output_image_path: Optional[Path]
    cropped_plate_path: Optional[Path]
    message: str


@dataclass(slots=True)
class InMemoryPipelineResult:
    annotated_image: Optional[np.ndarray]
    plate_image: Optional[np.ndarray]
    ocr_image: Optional[np.ndarray]
    detected_text: str
    is_valid: bool
    message: str


def _save_outputs(
    detection: PlateDetectionResult,
    plate_for_ocr,
    detected_text: str,
    config: AppConfig,
    image_path: Path,
) -> tuple[Path, Path]:
    annotated = draw_plate_box(
        detection.debug_image,
        detection.bounding_box,
        detected_text or "PLATE",
    )
    output_image_path = config.output_dir / f"{image_path.stem}_annotated.jpg"
    cropped_plate_path = config.output_dir / f"{image_path.stem}_plate.jpg"

    save_image(output_image_path, annotated)
    save_image(cropped_plate_path, plate_for_ocr)
    return output_image_path, cropped_plate_path


def _build_message(detected_text: str, is_valid: bool) -> str:
    if not detected_text:
        return "Plate detected, but OCR did not return usable text."
    if is_valid:
        return "Plate detected and OCR text looks plausible."
    return "Plate detected, but OCR text may need threshold tuning."


def _annotate_detection(
    detection: PlateDetectionResult,
    detected_text: str,
) -> np.ndarray:
    return draw_plate_box(
        detection.debug_image,
        detection.bounding_box,
        detected_text or "PLATE",
    )


def process_image(image_path: Path, config: Optional[AppConfig] = None) -> PipelineResult:
    config = config or AppConfig()
    image = cv2.imread(str(image_path))

    if image is None:
        return PipelineResult(
            image_path=image_path,
            detected_text="",
            is_valid=False,
            output_image_path=None,
            cropped_plate_path=None,
            message="Could not load image. Check the file path and image format.",
        )

    detection = detect_plate(image, config)
    if detection is None:
        return PipelineResult(
            image_path=image_path,
            detected_text="",
            is_valid=False,
            output_image_path=None,
            cropped_plate_path=None,
            message="No likely license plate region was detected.",
        )

    plate_for_ocr = preprocess_plate_for_ocr(detection.text_roi, config)
    raw_text = extract_text_from_plate(plate_for_ocr, config)
    detected_text = clean_plate_text(raw_text)
    is_valid = is_plausible_plate(detected_text)
    detection.debug_image = _annotate_detection(detection, detected_text)
    output_image_path, cropped_plate_path = _save_outputs(
        detection,
        plate_for_ocr,
        detected_text,
        config,
        image_path,
    )

    message = _build_message(detected_text, is_valid)

    return PipelineResult(
        image_path=image_path,
        detected_text=detected_text,
        is_valid=is_valid,
        output_image_path=output_image_path,
        cropped_plate_path=cropped_plate_path,
        message=message,
    )


def process_image_array(
    image: np.ndarray,
    config: Optional[AppConfig] = None,
) -> InMemoryPipelineResult:
    """Run the existing pipeline on an in-memory image for web apps."""

    config = config or AppConfig()
    if image.size == 0:
        return InMemoryPipelineResult(
            annotated_image=None,
            plate_image=None,
            ocr_image=None,
            detected_text="",
            is_valid=False,
            message="Uploaded image is empty.",
        )

    detection = detect_plate(image, config)
    if detection is None:
        return InMemoryPipelineResult(
            annotated_image=None,
            plate_image=None,
            ocr_image=None,
            detected_text="",
            is_valid=False,
            message="No likely license plate region was detected.",
        )

    plate_for_ocr = preprocess_plate_for_ocr(detection.text_roi, config)
    raw_text = extract_text_from_plate(plate_for_ocr, config)
    detected_text = clean_plate_text(raw_text)
    is_valid = is_plausible_plate(detected_text)
    annotated_image = _annotate_detection(detection, detected_text)
    message = _build_message(detected_text, is_valid)

    return InMemoryPipelineResult(
        annotated_image=annotated_image,
        plate_image=detection.text_roi,
        ocr_image=plate_for_ocr,
        detected_text=detected_text,
        is_valid=is_valid,
        message=message,
    )
