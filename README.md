Crop Disease Identification using Deep Learning

Plant Disease Classification (38 Classes)

Overview

This project implements a Convolutional Neural Network (CNN) model utilizing Transfer Learning (MobileNetV2) to accurately classify 38 different categories of healthy and diseased plant leaves. The core objective was to develop a high-performance classification model and resolve key MLOps challenges related to model deployment readiness.

The model achieved approximately 98% Validation Accuracy.

Key Features

38-Class Classification: Trained on the challenging PlantVillage dataset, covering 14 plant species and various healthy/disease states.

Transfer Learning: Uses the MobileNetV2 architecture as a base for efficient feature extraction.

High Accuracy: Demonstrates robust performance with high recall and precision across all classes (evidenced by the confusion matrix).

MLOps Efficiency Fix: Includes a separate utility script to strip data augmentation layers and serialize only the trained weights into a clean, production-ready model architecture.

Technical Stack

Core Language: Python 3.x

Frameworks: TensorFlow, Keras

Libraries: NumPy, Matplotlib, Scikit-learn, Seaborn

Deployment Target: Clean HDF5 weight file (.weights.h5) for inference.

Project Structure

.
├── data/
│   ├── train/        # Split data for training (80%)
│   └── val/          # Split data for validation (20%)
├── plantvillage dataset/  # Original raw dataset
├── notebook.ipynb    # Main development and training notebook (5-hour training cycle)
├── deployment_model_fix.py # Utility script to clean the model architecture
├── best_model.h5     # Model checkpoint saved during training (includes augmentation layers)
└── deployment_model_clean.weights.h5 # Final, production-ready model weights
└── README.md         # This file


Setup and Reproduction

To run and verify the model, follow these steps:

1. Environment Setup

It is highly recommended to use a virtual environment.

# Create and activate environment (example for Conda or venv)
conda create -n plant-ai python=3.9
conda activate plant-ai

# Install dependencies (ensure you are using versions compatible with Keras 3/TensorFlow 2.x)
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn split-folders


2. Data Preparation

The notebook.ipynb assumes the PlantVillage dataset is downloaded and structured appropriately.

Place the raw image data in the plantvillage dataset/color directory.

Run the initial cells in notebook.ipynb to execute the splitfolders.ratio command, which will automatically create the required data/train and data/val directories.

3. Model Training & Evaluation

The full model training procedure is contained within notebook.ipynb.

Run the notebook sequentially to load the data, define the MobileNetV2 architecture, perform transfer learning, and fine-tune the model.

This process generates the confusion matrix and saves the final trained model (e.g., best_model.h5).

