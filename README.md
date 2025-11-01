# 🌿 Crop Disease Identification using Deep Learning (MobileNetV2)

Can deep learning help farmers detect crop diseases early — just from a single photo of a leaf?

This project explores that question using the **PlantVillage dataset**, applying **TensorFlow** and **MobileNetV2** to accurately classify **38 types of healthy and diseased leaves**.  
The model achieves high accuracy after fine-tuning, and demonstrates how AI can make agriculture more efficient and sustainable.

---

## 🚀 Project Overview

This project aims to **identify crop diseases from leaf images** using a **convolutional neural network (CNN)** built with TensorFlow and Keras.  
By training on thousands of real-world images, the model learns to recognize subtle texture and color variations that indicate disease.

---

## 📂 Dataset

**Dataset Used:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- Total Images: **~50,000+**  
- Classes: **38 (Healthy + Diseased)**  
- Type: RGB Color Images  
- Split into:  
  - **Training Set:** 80%  
  - **Validation Set:** 20%

Each folder corresponds to a crop–disease class (e.g. `Tomato_Bacterial_spot`, `Potato_Early_blight`, `Apple_healthy`).

---

## ⚙️ Methodology

### 🧩 1. Data Preparation
- Loaded the dataset using `tf.keras.utils.image_dataset_from_directory()`
- Ensured class balance between training and validation
- Applied **data augmentation** to improve generalization:
  - Random rotations
  - Horizontal/vertical flips
  - Contrast normalization

### 🧠 2. Model Architecture
- **Base Model:** `MobileNetV2` (pretrained on ImageNet)
- **Top Layers Added:**
  - GlobalAveragePooling2D
  - Dense(512, activation='relu')
  - Dropout(0.3)
  - Dense(38, activation='softmax')

### ⚡ 3. Training Details
- **Loss Function:** `categorical_crossentropy`
- **Optimizer:** `Adam` with learning rate `0.0001`
- **Epochs:** Multiple phases — base training + fine-tuning
- **Fine-tuning:** Unfroze layers after index 100
- **Training Time:** ~5 hours on GPU

### 📈 4. Results
- High validation accuracy with minimal overfitting
- Smooth convergence observed during fine-tuning
- Robust performance across multiple crop categories

---
