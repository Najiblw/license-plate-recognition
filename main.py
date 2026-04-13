from __future__ import annotations

import argparse
from pathlib import Path

from src.config import AppConfig
from src.main import process_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="License plate recognition with Python, OpenCV, and Tesseract OCR."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input vehicle image.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default=None,
        help="Optional full path to tesseract.exe on Windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig()

    if args.tesseract_cmd:
        config.tesseract_cmd = args.tesseract_cmd

    result = process_image(Path(args.image), config)

    print(f"Image: {result.image_path}")
    print(f"Detected text: {result.detected_text or 'N/A'}")
    print(f"Plausible plate: {result.is_valid}")
    print(result.message)

    if result.output_image_path:
        print(f"Annotated output: {result.output_image_path}")
    if result.cropped_plate_path:
        print(f"Plate crop: {result.cropped_plate_path}")


if __name__ == "__main__":
    main()
