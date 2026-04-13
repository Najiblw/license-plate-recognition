from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import AppConfig
from .preprocess import preprocess_for_plate_detection


@dataclass(slots=True)
class PlateDetectionResult:
    text_roi: np.ndarray
    contour: np.ndarray
    bounding_box: tuple[int, int, int, int]
    debug_image: np.ndarray
    score: float


def _score_plate_candidate(
    contour: np.ndarray,
    edged: np.ndarray,
    image_area: int,
    config: AppConfig,
) -> tuple[float, np.ndarray, tuple[int, int, int, int]] | None:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) < 4 or len(approx) > 6:
        return None

    rect = cv2.minAreaRect(contour)
    (_, _), (rect_width, rect_height), _ = rect
    if rect_width <= 0 or rect_height <= 0:
        return None

    aspect_ratio = max(rect_width, rect_height) / min(rect_width, rect_height)
    rect_area = rect_width * rect_height
    area_ratio = rect_area / float(image_area)
    if not (
        config.min_plate_area_ratio <= area_ratio <= config.max_plate_area_ratio
        and config.min_plate_aspect_ratio <= aspect_ratio <= config.max_plate_aspect_ratio
    ):
        return None

    x, y, width, height = cv2.boundingRect(approx)
    if width == 0 or height == 0:
        return None

    bounding_aspect_ratio = width / float(height)
    if width <= height:
        return None
    if not (
        config.min_plate_aspect_ratio
        <= bounding_aspect_ratio
        <= config.max_plate_aspect_ratio
    ):
        return None

    contour_area = cv2.contourArea(contour)
    rectangularity = contour_area / float(rect_area)
    if rectangularity < config.min_rectangularity:
        return None

    edge_roi = edged[y : y + height, x : x + width]
    edge_density = cv2.countNonZero(edge_roi) / float(width * height)
    if edge_density < config.min_edge_density:
        return None

    aspect_score = 1.0 - min(
        abs(bounding_aspect_ratio - config.ideal_plate_aspect_ratio)
        / config.ideal_plate_aspect_ratio,
        1.0,
    )
    area_midpoint = (
        config.min_plate_area_ratio + config.max_plate_area_ratio
    ) / 2.0
    area_half_span = max(
        (config.max_plate_area_ratio - config.min_plate_area_ratio) / 2.0,
        1e-6,
    )
    area_score = 1.0 - min(abs(area_ratio - area_midpoint) / area_half_span, 1.0)
    rectangularity_score = min(rectangularity, 1.0)
    edge_score = min(edge_density / 0.15, 1.0)
    corner_bonus = 1.0 if len(approx) == 4 else 0.85
    score = (
        0.40 * aspect_score
        + 0.20 * area_score
        + 0.25 * rectangularity_score
        + 0.15 * edge_score
    ) * corner_bonus

    candidate_contour = approx
    return score, candidate_contour, (x, y, width, height)


def detect_plate(
    image: np.ndarray,
    config: AppConfig,
) -> Optional[PlateDetectionResult]:
    resized, _, edged = preprocess_for_plate_detection(image, config)
    contours, _ = cv2.findContours(
        edged.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[
        : config.max_candidate_contours
    ]

    image_area = resized.shape[0] * resized.shape[1]
    debug_image = resized.copy()

    best_candidate: Optional[PlateDetectionResult] = None
    best_score = -1.0

    for contour in contours:
        scored_candidate = _score_plate_candidate(contour, edged, image_area, config)
        if scored_candidate is None:
            continue

        score, candidate_contour, (x, y, width, height) = scored_candidate
        roi = resized[y : y + height, x : x + width]

        if roi.size == 0 or score <= best_score:
            continue

        best_score = score
        best_candidate = PlateDetectionResult(
            text_roi=roi,
            contour=candidate_contour,
            bounding_box=(x, y, width, height),
            debug_image=debug_image.copy(),
            score=score,
        )

    if best_candidate is None:
        return None

    cv2.drawContours(
        best_candidate.debug_image,
        [best_candidate.contour],
        -1,
        (0, 255, 0),
        2,
    )
    return best_candidate
