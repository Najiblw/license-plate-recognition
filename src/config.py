from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    """Central project settings for the recognition pipeline."""

    base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    input_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    samples_dir: Path = field(init=False)

    tesseract_cmd: str = field(
        default_factory=lambda: os.getenv(
            "TESSERACT_CMD",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        )
    )
    resize_width: int = 900
    canny_threshold_1: int = 40
    canny_threshold_2: int = 160
    clahe_clip_limit: float = 2.5
    clahe_grid_size: tuple[int, int] = (8, 8)
    adaptive_block_size: int = 31
    adaptive_c: int = 9
    morphology_kernel_size: int = 3
    ocr_scale_factor: float = 3.0
    ocr_lower_region_start: float = 0.35
    max_candidate_contours: int = 30
    min_plate_area_ratio: float = 0.01
    max_plate_area_ratio: float = 0.70
    min_plate_aspect_ratio: float = 2.0
    max_plate_aspect_ratio: float = 6.5
    ideal_plate_aspect_ratio: float = 2.8
    min_rectangularity: float = 0.35
    min_edge_density: float = 0.01
    ocr_psm: int = 7
    ocr_oem: int = 3
    ocr_whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def __post_init__(self) -> None:
        self.input_dir = self.base_dir / "data" / "input"
        self.output_dir = self.base_dir / "data" / "output"
        self.samples_dir = self.base_dir / "data" / "samples"


def build_tesseract_config(config: AppConfig) -> str:
    return (
        f"--oem {config.ocr_oem} "
        f"--psm {config.ocr_psm} "
        f"-c tessedit_char_whitelist={config.ocr_whitelist}"
    )
