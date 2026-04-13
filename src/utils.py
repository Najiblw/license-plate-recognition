from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def draw_plate_box(
    image: np.ndarray,
    bounding_box: tuple[int, int, int, int],
    label: str,
) -> np.ndarray:
    output = image.copy()
    x, y, width, height = bounding_box
    cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(
        output,
        label,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return output


def save_image(path: Path, image: np.ndarray) -> None:
    ensure_directory(path.parent)
    cv2.imwrite(str(path), image)
