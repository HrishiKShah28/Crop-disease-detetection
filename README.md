Crop Disease Classification (38 Classes)
Overview
This project implements a Convolutional Neural Network (CNN) model utilizing Transfer Learning (MobileNetV2) to accurately classify 38 different categories of healthy and diseased plant leaves from the PlantVillage dataset.

The core achievement was not just the high accuracy (approx. 98% Validation Accuracy), but also successfully resolving a critical MLOps challenge related to model deployment readiness.

Key Features
High Accuracy: Achieved ~98% Validation Accuracy on a challenging, highly multi-class dataset.

Transfer Learning: Uses MobileNetV2 for efficient feature extraction, ensuring a balance between performance and computational efficiency.

MLOps Efficiency Fix: Successfully developed and implemented a weight-transfer process to strip training-only layers (like data augmentation) from the saved model, creating a clean, production-ready inference file (.weights.h5) without requiring a full 5-hour re-training cycle.

Technical Stack
Core Language: Python 3.x

Frameworks: TensorFlow, Keras

Libraries: NumPy, Matplotlib, Scikit-learn, Seaborn

1. Environment Setup
Bash

# Install dependencies (ensure compatibility with TensorFlow 2.x / Keras)
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn split-folders
2. Data and Training
Place the raw PlantVillage data in the appropriate directory (plantvillage dataset/color).

Run the notebook.ipynb sequentially to perform the data split, define the MobileNetV2 architecture, execute transfer learning, and fine-tune the model. This generates the performance plots and the confusion matrix.
