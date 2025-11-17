📄 Smart OMR Sheet Scanner Using OpenCV

A simple computer vision project that scans an OMR sheet and automatically detects filled bubbles using OpenCV, image processing, and a classification-based approach.

🚀 Features

Detects and extracts bubbles from the OMR sheet

Classifies each bubble as filled or unfilled

Compares detected answers with an answer key

Calculates the final score

Works with mobile phone images or scanned sheets

🔧 Technologies Used

Python

OpenCV

NumPy

Imutils

Machine Learning (SVM / Logistic Regression / CNN)

🛠 How It Works

Preprocess image (grayscale, blur, edge detection)

Apply perspective transform to align the sheet

Detect and filter bubble contours

Use a classifier to check if a bubble is filled

Match with answer key and calculate score.
