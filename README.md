# License Plate Recognition Starter

This project is a simple starter template for building a license plate recognition system with Python, OpenCV, and Tesseract OCR.

## What This Version Does

- Loads a single image from disk
- Detects a likely license plate region with OpenCV contours
- Preprocesses the plate crop for OCR
- Extracts text with Tesseract OCR
- Saves an annotated image and cropped plate output

## Project Structure

```text
license-plate-recognition/
├── data/
│   ├── input/
│   ├── output/
│   └── samples/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── ocr.py
│   ├── plate_detector.py
│   ├── postprocess.py
│   ├── preprocess.py
│   └── utils.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── README.md
└── run.py
```

## Install OpenCV and pytesseract

1. Install the Python packages:

```bash
pip install -r requirements.txt
```

If you want to install only the core OCR packages first, use:

```bash
pip install opencv-python pytesseract
```

2. Install the Tesseract OCR application itself on Windows.

The Python package `pytesseract` is only a wrapper, so you still need the Tesseract executable installed on your machine. After installation, the common path is:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

3. If Tesseract is installed in a different folder, pass the path when running the project:

```bash
python main.py --image data/input/car.jpg --tesseract-cmd "C:\Path\To\tesseract.exe"
```

## Run in Anaconda Jupyter Notebook

Use this when you want to work on the project as a notebook-based machine learning workflow instead of only from the command line.

1. Create the conda environment:

```bash
conda env create -f environment.yml
```

2. Activate it:

```bash
conda activate license-plate-ocr
```

3. Start Jupyter:

```bash
jupyter lab
```

4. Open the starter notebook:

```text
notebooks/license_plate_recognition.ipynb
```

5. In the notebook, set:

- `IMAGE_PATH` to the image you want to test
- `TESSERACT_CMD` to your local Tesseract executable path

Important:

- In Jupyter, use inline display with matplotlib rather than `cv2.imshow`
- The notebook already calls the shared pipeline in `src/main.py`, so you are not duplicating project logic
- If you launch Jupyter from the `notebooks/` folder, the notebook automatically adds the project root to `sys.path`

## Step-by-Step Implementation Plan

1. Put a test image into `data/input/`.
2. Run the pipeline on one image.
3. Inspect the saved plate crop in `data/output/`.
4. Tune contour and OCR settings in `src/config.py`.
5. Add more images with different lighting and angles.
6. Improve OCR cleanup rules in `src/postprocess.py`.
7. Replace contour-based detection with a trained detector later if needed.

## How To Run

Example:

```bash
python main.py --image data/input/car.jpg
```

The script prints:

- detected text
- whether the text looks like a plausible plate
- where outputs were saved

## File Responsibilities

- `src/config.py`: central settings, OCR options, folder paths
- `src/preprocess.py`: image resizing, filtering, thresholding
- `src/plate_detector.py`: plate candidate detection from contours
- `src/ocr.py`: Tesseract configuration and OCR call
- `src/postprocess.py`: OCR text cleanup and validation
- `src/main.py`: complete pipeline for one image
- `src/utils.py`: saving images and drawing annotations
- `main.py`: command-line entry point
- `run.py`: alternate entry point kept for convenience

## Next Improvements

- Add webcam or video support
- Log confidence scores and debug images
- Use a deep learning detector such as YOLO for better accuracy
- Add country-specific plate validation rules
