#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py

import os
import uuid
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Image processing
import cv2
import numpy as np
from PIL import Image

# Deep learning models
import torch
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Text extraction and NLP
import pytesseract
from transformers import pipeline

# Metrics calculation
from sklearn.metrics import precision_score, recall_score, f1_score

# Data handling
import json

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/uploads'
app.config['SEGMENT_FOLDER'] = 'static/images/segments'
app.config['OUTPUT_FOLDER'] = 'static/images/output'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SEGMENT_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load YOLOv5 model
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load Mask R-CNN model
model_maskrcnn = maskrcnn_resnet50_fpn(pretrained=True)
model_maskrcnn.eval()

# Load T5 summarization model
summarizer = pipeline("summarization", model="t5-large")

# Define route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save uploaded image
            filename = secure_filename(file.filename)
            master_id = str(uuid.uuid4())
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{master_id}_{filename}")
            file.save(filepath)

            # Run the pipeline
            result_data, original_image, annotated_image_path, accuracy_metrics = run_pipeline(filepath, master_id)

            # Render the result template
            return render_template('result.html',
                                   original_image=original_image,
                                   annotated_image=annotated_image_path,
                                   result_data=result_data,
                                   accuracy=accuracy_metrics)
    return render_template('index.html')

def run_pipeline(image_path, master_id):
    # Step 1: Image Segmentation
    segments, segment_paths, boxes = segment_image(image_path, master_id)

    # Step 2: Object Detection and Description
    detections = detect_objects(segment_paths, master_id)

    # Step 3: Text Extraction
    detections = extract_text(detections)

    # Step 4: Summarization
    detections = summarize_attributes(detections)

    # Step 5: Data Mapping and Accuracy Calculation
    # Load ground truth labels (if available)
    ground_truth = load_ground_truth(master_id)
    accuracy_metrics = calculate_accuracy(detections, ground_truth)

    # Step 6: Generate Annotated Image
    annotated_image_path = create_annotated_image(image_path, detections, boxes, master_id)

    # Prepare data for display
    result_data = []
    for det in detections:
        result_data.append({
            'id': det['id'],
            'master_id': det['master_id'],
            'segment_image': det['segment_path'],
            'description': det['description'],
            'text': det['text'],
            'summary': det['summary']
        })

    return result_data, image_path, annotated_image_path, accuracy_metrics

def segment_image(image_path, master_id, threshold=0.8):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        outputs = model_maskrcnn(image_tensor)

    # Get masks, scores, and boxes
    masks = outputs[0]['masks']
    scores = outputs[0]['scores']
    boxes = outputs[0]['boxes']

    # Filter out low scoring masks
    masks = masks[scores > threshold]
    boxes = boxes[scores > threshold]

    segment_paths = []
    for idx, mask in enumerate(masks):
        mask = mask.squeeze().cpu().numpy()
        binary_mask = (mask > 0.5).astype(np.uint8)

        # Extract object using mask
        np_image = np.array(image)
        segmented = cv2.bitwise_and(np_image, np_image, mask=binary_mask)

        # Save segmented image
        segment_id = str(uuid.uuid4())
        segment_path = os.path.join(app.config['SEGMENT_FOLDER'], f"{segment_id}.png")
        cv2.imwrite(segment_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))

        segment_paths.append(segment_path)
    return masks, segment_paths, boxes

def detect_objects(segment_paths, master_id):
    detections = []
    for segment_path in segment_paths:
        image = Image.open(segment_path).convert("RGB")
        results = model_yolo(segment_path)

        # Get predictions
        labels = results.xyxyn[0][:, -1]
        if len(labels) == 0:
            description = 'Unknown'
        else:
            description = model_yolo.names[int(labels[0])]

        detection = {
            'id': os.path.splitext(os.path.basename(segment_path))[0],
            'master_id': master_id,
            'segment_path': segment_path,
            'description': description,
            'text': ''
        }
        detections.append(detection)
    return detections

def extract_text(detections):
    for det in detections:
        image = Image.open(det['segment_path']).convert('RGB')
        # Preprocess image for OCR
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        # Apply thresholding to clean the image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Use pytesseract to extract text
        text = pytesseract.image_to_string(thresh)
        det['text'] = text.strip()
    return detections

def summarize_attributes(detections):
    for det in detections:
        content = f"{det['description']}. {det['text']}"
        if len(content.strip()) == 0:
            summary = "No significant attributes."
        else:
            try:
                summary_result = summarizer(content, max_length=50, min_length=5, do_sample=False)
                summary = summary_result[0]['summary_text']
            except Exception:
                summary = content  # Fallback if summarization fails
        det['summary'] = summary
    return detections

def calculate_accuracy(detections, ground_truth):
    if not ground_truth:
        return {'precision': 'N/A', 'recall': 'N/A', 'f1_score': 'N/A'}
    y_true = []
    y_pred = []
    for det in detections:
        pred_label = det['description']
        true_label = ground_truth.get(det['id'], 'Unknown')
        y_pred.append(pred_label)
        y_true.append(true_label)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy_metrics = {
        'precision': f"{precision:.2f}",
        'recall': f"{recall:.2f}",
        'f1_score': f"{f1:.2f}"
    }
    return accuracy_metrics

def load_ground_truth(master_id):
    # Load ground truth labels from a JSON file or database
    # For demonstration, return an empty dictionary
    # You can implement this function to load actual ground truth labels
    ground_truth = {}
    # Example:
    # ground_truth = {'segment_id1': 'car', 'segment_id2': 'person', ...}
    return ground_truth

def create_annotated_image(image_path, detections, boxes, master_id):
    # Load the original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for det, box in zip(detections, boxes):
        x1, y1, x2, y2 = box.int()
        label = det['description']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    annotated_image_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{master_id}_annotated.jpg")
    cv2.imwrite(annotated_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return annotated_image_path

# Templates

# Ensure the templates directory exists
os.makedirs('templates', exist_ok=True)

# templates/index.html
index_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Pipeline - Upload Image</title>
</head>
<body>
    <h1>Upload an Image</h1>
    <form action="/" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <br><br>
        <input type="submit" value="Process Image">
    </form>
</body>
</html>
'''

# templates/result.html
result_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Pipeline - Results</title>
</head>
<body>
    <h1>Processing Results</h1>
    <h2>Original Image</h2>
    <img src="{{ url_for('static', filename=original_image.split('static/')[-1]) }}" alt="Original Image" width="500">

    <h2>Annotated Image with Segments</h2>
    <img src="{{ url_for('static', filename=annotated_image.split('static/')[-1]) }}" alt="Annotated Image" width="500">

    <h2>Accuracy Metrics</h2>
    <p>Precision: {{ accuracy['precision'] }}</p>
    <p>Recall: {{ accuracy['recall'] }}</p>
    <p>F1-Score: {{ accuracy['f1_score'] }}</p>

    <h2>Data Table</h2>
    <table border="1">
        <tr>
            <th>Segment Image</th>
            <th>ID</th>
            <th>Description</th>
            <th>Extracted Text</th>
            <th>Summary</th>
        </tr>
        {% for obj in result_data %}
        <tr>
            <td><img src="{{ url_for('static', filename=obj['segment_image'].split('static/')[-1]) }}" alt="Segment Image" width="200"></td>
            <td>{{ obj['id'] }}</td>
            <td>{{ obj['description'] }}</td>
            <td>{{ obj['text'] }}</td>
            <td>{{ obj['summary'] }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
'''

# Save the templates
with open('templates/index.html', 'w') as f:
    f.write(index_html)
with open('templates/result.html', 'w') as f:
    f.write(result_html)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

