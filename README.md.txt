# MRI Brain Tumor Classification and Segmentation — Ensemble Deep Learning System

## Overview

This project implements an ensemble deep learning system for MRI-based brain tumor analysis. It performs:

• Tumor classification using multiple transformer and CNN backbones
• Tumor segmentation using U-Net architecture
• Ensemble prediction for improved robustness and accuracy

The system is designed for medical image decision support and explainable AI workflows.

---

## Models Used

### Classification Ensemble

* ResNet18 (CNN backbone)
* Vision Transformer — Tiny (ViT-Tiny)
* Swin Transformer — Tiny

Predictions from all three models are combined using ensemble voting / probability averaging.

### Segmentation

* U-Net architecture for tumor region segmentation

---

## Features

* Multi-model ensemble classification
* Transformer + CNN hybrid approach
* MRI tumor segmentation mask generation
* Flask-based inference interface
* Modular model loading pipeline
* Ready for Grad-CAM / explainability integration

---

## Tech Stack

Python
PyTorch
Flask
OpenCV
NumPy
Deep Learning (CNN + Transformers)

---

## Model Weights

Trained weights are not included due to file size limits.
Models were trained in Kaggle environment.

To run locally:

1. Place trained model files inside `/models`
2. Update model paths in code
3. Run the Flask app

---

## How to Run

```
pip install -r requirements.txt
python app.py
```

Open browser → [http://localhost:5000](http://localhost:5000)

---

## Project Type

Medical Imaging AI — Classification + Segmentation Ensemble System

---

## Author

Sivasankari S
