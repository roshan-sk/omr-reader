# OMR Sheet Detector

This project uses OpenCV to process and detect answers from OMR (Optical Mark Recognition) bubble sheets. The program reads an image of the OMR sheet, processes it to detect bubbles, and extracts answers for each question.


## How it Works

1. Image Preprocessing:
The input image is resized, converted to grayscale, and blurred to remove noise. Then, edge detection is applied to find the
contour of the OMR sheet.

2. Perspective Transformation:
If the OMR sheet is detected, the image is warped to a top-down view using perspective transformation.

3. Bubble Detection:
The image is divided into columns, and each column is processed to detect the bubbles for the questions. The number of detected bubbles is counted to determine the selected answer.

4. Answer Extraction:
For each row (question) in the OMR sheet, the bubbles are analyzed, and the option with the most marked pixels is considered as the answer.


## Features
- Detects OMR sheet
- Warps perspective
- Extracts answers automatically
- Supports multi-column bubble sheets

## Tech Stack
- Python
- OpenCV
- NumPy

##  Output Video
Video URL : https://drive.google.com/file/d/1savHVyKfyZeCmYuceX_cegZcv7EQ0SR9/view?usp=drive_link

## How to Run

```bash
pip install -r requirements.txt
python omr_detector.py

