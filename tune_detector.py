from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2

from src.config import AppConfig
from src.plate_detector import detect_plate


def iter_images(input_dir: Path) -> list[Path]:
    images: list[Path] = []
    seen: set[str] = set()
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for path in sorted(input_dir.glob(pattern)):
            normalized = str(path.resolve()).lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            images.append(path)
    return images


def main() -> None:
    base_config = AppConfig()
    images = iter_images(base_config.input_dir)
    if not images:
        print("No input images found.")
        return

    configs = []
    for canny_1, canny_2 in ((40, 160), (50, 200), (70, 220)):
        for min_rectangularity in (0.35, 0.45, 0.55):
            for min_edge_density in (0.01, 0.02, 0.03):
                for min_plate_area_ratio in (0.005, 0.01):
                    configs.append(
                        replace(
                            base_config,
                            canny_threshold_1=canny_1,
                            canny_threshold_2=canny_2,
                            min_rectangularity=min_rectangularity,
                            min_edge_density=min_edge_density,
                            min_plate_area_ratio=min_plate_area_ratio,
                        )
                    )

    best_config = None
    best_hits = -1
    best_score_total = -1.0
    best_details: list[str] = []

    for config in configs:
        hits = 0
        score_total = 0.0
        details: list[str] = []

        for image_path in images:
            image = cv2.imread(str(image_path))
            detection = detect_plate(image, config) if image is not None else None
            if detection is None:
                details.append(f"{image_path.name}: no detection")
                continue

            hits += 1
            score_total += detection.score
            details.append(
                f"{image_path.name}: detection score={detection.score:.3f}, "
                f"box={detection.bounding_box}"
            )

        if hits > best_hits or (hits == best_hits and score_total > best_score_total):
            best_hits = hits
            best_score_total = score_total
            best_config = config
            best_details = details

    print("Best detector configuration:")
    print(
        f"canny=({best_config.canny_threshold_1}, {best_config.canny_threshold_2}), "
        f"min_rectangularity={best_config.min_rectangularity}, "
        f"min_edge_density={best_config.min_edge_density}, "
        f"min_plate_area_ratio={best_config.min_plate_area_ratio}"
    )
    print(f"Detections: {best_hits}/{len(images)}")
    print(f"Total detection score: {best_score_total:.3f}")
    for detail in best_details:
        print(detail)


if __name__ == "__main__":
    main()
