# AI Pipeline for Image Segmentation and Object Analysis

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
  - [Step 1: Image Segmentation](#step-1-image-segmentation)
  - [Step 2: Object Detection and Description](#step-2-object-detection-and-description)
  - [Step 3: Text Extraction](#step-3-text-extraction)
  - [Step 4: Summarization](#step-4-summarization)
  - [Step 5: Data Mapping and Accuracy Calculation](#step-5-data-mapping-and-accuracy-calculation)
  - [Step 6: Output Generation](#step-6-output-generation)
- [Accuracy Metrics](#accuracy-metrics)
- [Performance and Efficiency](#performance-and-efficiency)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

This project implements an AI pipeline that processes an input image to segment, identify, and analyze objects within the image. The output is a summary table with mapped data for each object and an annotated image.

## Features

- **Image Segmentation**: Segments objects from the input image using Mask R-CNN.
- **Object Detection**: Identifies objects using YOLOv5 for higher accuracy.
- **Text Extraction**: Extracts text from each segmented object using Tesseract OCR.
- **Summarization**: Generates a summary of the object's attributes using BART NLP model.
- **Web Interface**: Allows users to upload images and view results via a Flask web application.
- **Accuracy Metrics**: Calculates and displays accuracy metrics like precision, recall, and F1-score.
- **Efficient Processing**: Optimized code to handle complex images efficiently.

## Directory Structure

ai-pipeline/
├── app.py
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   ├── images/
│   │   ├── uploads/
│   │   └── segments/
├── README.md
└── requirements.txt

## Usage 

•Uploading an Image
•Click on the "Choose File" button to select an image from your computer.
•Accepted formats: JPG, PNG, JPEG, BMP.
•Click on "Process Image" to start the analysis.

## Viewing Results

•After processing, the results page will display:
•Original Image: The image you uploaded.
•Accuracy Metrics: Precision, Recall, and F1-Score (placeholders).
•Data Table: Contains segmented images, IDs, descriptions, extracted text, and summaries.

## Pipeline Steps

Step 1: Image Segmentation

Model: Mask R-CNN (pretrained on COCO dataset).
Process:
The uploaded image is segmented to identify different objects.
Segmentation masks are applied to extract individual objects.
Segmented images are saved for further analysis.

Step 2: Object Detection and Description

Model: YOLOv5s (pretrained).
Process:
Each segmented image is passed through YOLOv5 for object detection.
The model predicts the class of the object.
If no object is detected, it is labeled as "Unknown".

Step 3: Text Extraction

Tool: Tesseract OCR.
Process:
Extracts text from each segmented image.
Useful for objects containing text (e.g., signs, documents).

Step 4: Summarization

Model: BART Large CNN.
Process:
Combines the object description and extracted text.
Generates a concise summary of the object's attributes.

Step 5: Data Mapping and Accuracy Calculation

Process:
Each object's data is compiled into a structured format.
Accuracy metrics are calculated (currently placeholders).

Step 6: Output Generation

Process:
Results are displayed on a web page.
Includes the original image, accuracy metrics, and a data table.

## Accuracy Metrics

Precision: Not applicable without ground truth labels.
Recall: Not applicable without ground truth labels.
F1-Score: Not applicable without ground truth labels.
Note: To calculate actual accuracy metrics, you need a dataset with ground truth annotations.

## Performance and Efficiency

Optimized Models: Using YOLOv5s and BART Large CNN for a balance between speed and accuracy.
Efficient Code: Single script implementation for ease of testing and deployment.

Project Structure

ai-pipeline/
├── app.py
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   ├── images/
│   │   ├── uploads/
│   │   └── segments/
├── README.md
└── requirements.txt

## Dependencies

Flask: Web framework for the application.

Torch and Torchvision: For deep learning models.

Ultralytics YOLOv5: For object detection.

Pillow (PIL): Image processing.

OpenCV: Image processing.

Pytesseract: OCR for text extraction.

Transformers: For NLP tasks (summarization).

Matplotlib: For image visualization (if needed).


## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.

## License
This project is licensed under the MIT License.

## Contact
For any questions or support, please contact:

Email: utkarshtiwar89@gmail.com
LinkedIn: https://www.linkedin.com/in/utkarsh-tiwari-174212216/
