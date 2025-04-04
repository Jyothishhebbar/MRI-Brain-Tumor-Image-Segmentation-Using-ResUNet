# MRI Brain Tumor Segmentation Using ResUNet

This project focuses on segmenting brain tumors from MRI scans using a deep learning model based on the **ResUNet** architecture. ResUNet combines the segmentation strength of **U-Net** with the learning efficiency of **residual connections**, providing accurate and automated tumor segmentation.

## 📌 Objective

- Develop a ResUNet-based model for brain tumor segmentation.
- Preprocess MRI data for better generalization.
- Train using the **LGG MRI Segmentation Dataset** from Kaggle.
- Optimize using loss functions like Dice Loss and Binary Cross-Entropy.
- Evaluate with Dice Coefficient and Intersection over Union (IoU).
- Minimize false positives/negatives for reliable predictions.

## ⚙️ Methodology

### Functional Requirements
- **Input**: Accept MRI images (PNG/JPEG).
- **Output**: Generate segmentation masks highlighting tumor regions.
- **Performance Metrics**: Dice Coefficient, IoU, accuracy.
- **Training/Evaluation**: Support model training and performance tracking.

### Non-Functional Requirements
- **Usability**: Intuitive UI for MRI input and result visualization.
- **Scalability**: Extendable to support more tumor types/features.
- **Performance**: Responsive even on large datasets.
- **Error Handling**: Robust handling of invalid input and training errors.

### Architecture
1. **Input Module** – Loads and preprocesses images.
2. **Model Module** – ResUNet implementation for segmentation.
3. **Output Module** – Displays results and performance metrics.

## 🛠 Implementation

- **Platform**: Google Colab with GPU.
- **Libraries**: TensorFlow/PyTorch, NumPy, OpenCV, Pandas, Matplotlib.
- **Data Preprocessing**:
  - Normalize pixel values.
  - Resize to 256×256 pixels.
  - Apply augmentation (rotate, flip, zoom).

### Model Development & Training
- Encoder-decoder architecture with skip and residual connections.
- Optimized with Adam optimizer and Dice Loss.
- Training monitored via TensorBoard.

### Evaluation
- Dice Coefficient and IoU for quantitative evaluation.
- Visual comparison with ground truth for qualitative analysis.

## ✅ Conclusion

The ResUNet model effectively segmented tumor regions from MRI scans with high accuracy. Its architecture enabled both detailed spatial feature extraction and robust global context understanding.

## 📚 References

- Dataset: [Kaggle - LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- [Feature Extraction Concepts](https://setosa.io/ev/image-kernels/)
- [CNN Visualization](https://www.cs.ryerson.ca/~aharley/vis/conv/flat.html)
- [Transfer Learning Article](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)

---

