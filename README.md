# 🧠 ImageNet Feature Extractor & Model Training (TensorFlow)

This project demonstrates training and evaluating a custom image classification model using **TensorFlow**, **Albumentations**, and **OpenCV**, built for working with ImageNet-like datasets.

---

### 📁 Project Structure

```
├── imagenet-new.ipynb        # Main notebook with training pipeline
├── requirements.txt          # Required Python dependencies
├── README.md                 # This documentation
```

---

### 🛠️ Installation

Ensure Python 3.7+ is installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
tensorflow==2.12.0
opencv-python
scikit-learn
matplotlib
albumentations==1.3.0
```

**For Colab/TPU (optional):**
```bash
!apt-get update && apt-get install -y python3-opencv
```

---

### 📦 Included Libraries

- `TensorFlow` for model building and training
- `Albumentations` for advanced data augmentation
- `OpenCV` for image manipulation
- `scikit-learn` for metrics and label encoding
- `matplotlib` for visualization
- `PIL`, `numpy`, and `pandas` for preprocessing

---

### 🧪 Features

- 📦 **Data Augmentation** using Albumentations
- 🔍 **Efficient Preprocessing** with OpenCV
- 🚀 **Model Acceleration** using TPU (optional)
- 📈 **Visualization and Debugging** with matplotlib

---

### 🚀 How to Use

1. Open `imagenet-new.ipynb` in Jupyter or Google Colab.
2. Modify the dataset path and augmentation strategy as needed.
3. Run all cells for full training and evaluation.
4. Monitor performance metrics and visual plots.

---

### ⚙️ Example Setup Code (From Notebook)

```python
import tensorflow as tf
from tensorflow import keras
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import warnings

warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

---

### 🔗 TPU Compatibility

The notebook detects TPU and connects if available:

```python
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None
```

---
