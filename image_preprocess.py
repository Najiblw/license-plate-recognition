from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pytesseract

from src.postprocess import clean_plate_text, is_plausible_plate


def find_plate_contour(edges):
    contours, _ = cv2.findContours(
        edges.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    best_candidate = None
    best_score = -1.0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) < 4 or len(approximation) > 6:
            continue

        rect = cv2.minAreaRect(contour)
        (_, _), (rect_width, rect_height), _ = rect
        if rect_width <= 0 or rect_height <= 0:
            continue

        x, y, width, height = cv2.boundingRect(approximation)
        if width == 0 or height == 0:
            continue

        aspect_ratio = max(rect_width, rect_height) / min(rect_width, rect_height)
        area = rect_width * rect_height
        bounding_aspect_ratio = width / float(height)
        rectangularity = cv2.contourArea(contour) / float(area)
        edge_density = cv2.countNonZero(edges[y : y + height, x : x + width]) / float(
            width * height
        )

        if width <= height:
            continue
        if not (
            2.0 <= aspect_ratio <= 6.5
            and 2.0 <= bounding_aspect_ratio <= 6.5
            and area > 1000
            and rectangularity >= 0.35
        ):
            continue
        if edge_density < 0.01:
            continue

        aspect_score = 1.0 - min(abs(bounding_aspect_ratio - 2.8) / 2.8, 1.0)
        score = (
            0.45 * aspect_score
            + 0.30 * min(rectangularity, 1.0)
            + 0.25 * min(edge_density / 0.15, 1.0)
        ) * (1.0 if len(approximation) == 4 else 0.85)

        if score > best_score:
            best_score = score
            best_candidate = (approximation, (x, y, width, height))

    if best_candidate is None:
        return None, None
    return best_candidate


def crop_plate_region(image, bounding_box):
    x, y, width, height = bounding_box
    return image[y : y + height, x : x + width]


def threshold_plate_for_ocr(plate_image):
    start_row = int(plate_image.shape[0] * 0.35)
    focused_region = plate_image[start_row:, :]
    if focused_region.size == 0:
        focused_region = plate_image

    plate_grayscale = cv2.cvtColor(focused_region, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(plate_grayscale)
    filtered = cv2.bilateralFilter(enhanced, 7, 25, 25)
    enlarged = cv2.resize(
        filtered,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC,
    )
    thresholded = cv2.adaptiveThreshold(
        enlarged,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def configure_tesseract(tesseract_cmd: str | None) -> None:
    if tesseract_cmd and Path(tesseract_cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def extract_plate_text(thresholded_plate) -> str:
    try:
        raw_text = pytesseract.image_to_string(
            thresholded_plate,
            config="--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
    except pytesseract.TesseractNotFoundError:
        return ""
    return clean_plate_text(raw_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an image, convert it to grayscale, blur it, run Canny edge "
            "detection, find a likely license plate contour, crop it, and "
            "threshold it for OCR."
        )
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--tesseract-cmd",
        default=None,
        help="Optional full path to tesseract.exe on Windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_tesseract(args.tesseract_cmd)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(grayscale)
    blurred = cv2.bilateralFilter(enhanced, 11, 17, 17)
    edges = cv2.Canny(blurred, 40, 160)
    plate_contour, bounding_box = find_plate_contour(edges)

    boxed_image = image.copy()
    cropped_plate = None
    thresholded_plate = None
    detected_text = ""

    if bounding_box is not None:
        x, y, width, height = bounding_box
        cv2.rectangle(
            boxed_image,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            boxed_image,
            "Likely Plate",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cropped_plate = crop_plate_region(image, bounding_box)
        thresholded_plate = threshold_plate_for_ocr(cropped_plate)
        detected_text = extract_plate_text(thresholded_plate)

    if plate_contour is not None:
        cv2.drawContours(boxed_image, [plate_contour], -1, (255, 0, 0), 2)

    if detected_text:
        print(f"Detected license number: {detected_text}")
        print(f"OCR output looks plausible: {is_plausible_plate(detected_text)}")
    elif bounding_box is not None:
        print("Detected license number: OCR did not return usable text.")
        print("If Tesseract is installed elsewhere, pass --tesseract-cmd with its full path.")
    else:
        print("Detected license number: No license plate candidate found.")

    cv2.imshow("Blurred Image", blurred)
    cv2.imshow("Canny Edges", edges)
    cv2.imshow("Detected Plate Region", boxed_image)

    if cropped_plate is not None and thresholded_plate is not None:
        cv2.imshow("Cropped Plate", cropped_plate)
        cv2.imshow("Thresholded Plate", thresholded_plate)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
