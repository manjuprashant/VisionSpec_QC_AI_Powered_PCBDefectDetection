# VisionSpec QC – AI-Powered PCB Defect Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

VisionSpec QC is an end-to-end deep learning system for **automated PCB defect detection** using Computer Vision.
It trains and compares multiple CNN and transfer learning models, generates explainable AI visualizations using Grad-CAM, and performs real-time inference.

This project simulates an **industrial Automated Optical Inspection (AOI) pipeline** used in electronics manufacturing.

---

# Project Demo Pipeline

```
PCB Images Dataset
        │
        ▼
Dataset Generator / Loader
        │
        ▼
Model Training
(CNN, MobileNetV2, ResNet50, EfficientNetB0)
        │
        ▼
Saved Models (.h5)
        │
        ├── Model Evaluation
        │      • Accuracy
        │      • Precision
        │      • Recall
        │      • F1 Score
        │      • ROC Curve
        │
        ├── Grad-CAM
        │      • Heatmaps
        │      • Explain predictions
        │
        └── Real-time Inference
               • Webcam
               • Dataset simulation
```

---

# Key Features

## Multi-Model Training

Train and compare:

• VisionSpecQC Custom CNN
• MobileNetV2
• ResNet50
• EfficientNetB0

---

## Model Evaluation Dashboard

Automatically generates:

• Accuracy comparison bar chart
• ROC curve
• Precision-Recall curve
• Confidence score boxplot
• CSV performance report

---

## Explainable AI (Grad-CAM)

Visualizes:

• Defect regions
• Model attention areas

Helps validate model reliability.

---

## Real-Time Inspection Simulation

Supports:

• Webcam inspection
• Dataset simulation
• Confidence score display

---

# Project Structure

```
VisionSpec_QC_GitHub_Project
│
├── dataset
│   ├── train
│   │   ├── DEFECT
│   │   └── OK
│   │
│   └── val
│       ├── DEFECT
│       └── OK
│
├── models
│   ├── visionspec_qc.h5
│   ├── mobilenetv2.h5
│   ├── resnet50.h5
│   └── efficientnetb0.h5
│
├── scripts
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── realtime_inference.py
│   └── realtime_dataset_inference.py
│
├── evaluation_outputs
├── gradcam_outputs
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```
git clone https://github.com/yourusername/VisionSpec_QC_GitHub_Project.git
cd VisionSpec_QC_GitHub_Project
```

---

## Create Virtual Environment

Windows

```
python -m venv venv
venv\Scripts\activate
```

Linux / Mac

```
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```
pip install -r requirements.txt
```

---

# Dataset Format

```
dataset/
    train/
        DEFECT/
        OK/

    val/
        DEFECT/
        OK/
```

Binary classification:

0 = DEFECT
1 = OK

---

# Train Models

```
python scripts/train.py
```

Output:

```
models/
```

Saved models:

```
visionspec_qc.h5
mobilenetv2.h5
resnet50.h5
efficientnetb0.h5
```

---

# Evaluate and Compare Models

```
python scripts/evaluate.py
```

Output:

```
evaluation_outputs/

metrics.csv
accuracy_bar_chart.png
roc_curve.png
precision_recall_curve.png
confidence_boxplot.png
```

---

# Generate Grad-CAM

```
python scripts/gradcam.py
```

Output:

```
gradcam_outputs/

train/
val/

gradcam_report_train.csv
gradcam_report_val.csv
```

CSV includes:

image path
true class
predicted class
confidence score

---

# Real-Time Webcam Inference

```
python scripts/realtime_inference.py
```

Press **Q** to exit.

---

# Dataset Real-Time Simulation

```
python scripts/realtime_dataset_inference.py
```

Shows:

Prediction
Confidence score
Grad-CAM

---

# Example Results

## Accuracy Comparison

Example:

VisionSpecQC CNN — 96%
MobileNetV2 — 95%
ResNet50 — 97%
EfficientNetB0 — 98%

Best model selected automatically.

---

# Explainable AI Example

Grad-CAM highlights defect areas.

Red regions = model focus area.

---

# Technologies Used

Python
TensorFlow / Keras
OpenCV
NumPy
Matplotlib
Scikit-learn

---

# Industrial Applications

Electronics manufacturing QA
PCB defect inspection
Automated optical inspection
Smart factories
Industrial AI systems

---

# Future Improvements

Deploy using Flask / FastAPI
Edge deployment (Jetson Nano)
YOLO defect localization
Web dashboard

---

# Author

Manjula Srinivasan
AI / Computer Vision Developer

---

# License

MIT License
