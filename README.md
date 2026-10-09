# Industrial Visual Anomaly Detection using ResNet18

Industrial anomaly detection system built using **PyTorch, pretrained ResNet18, and the MVTec AD dataset** to identify defective bottle images without using defective samples during training.

## Project Overview

The MVTec AD dataset provides only **normal images for training**, making this a one-class anomaly detection problem rather than a standard binary classification task.

A pretrained **ResNet18** model is used as a feature extractor. Each bottle image is converted into a **512-dimensional feature embedding**.

Normal training embeddings are stored as a reference feature bank. For a new image, the system calculates its distance from the closest normal training embedding.

A large distance indicates that the image is visually different from known normal samples and is therefore considered anomalous.

## Pipeline

```text
Bottle Image
    ↓
Resize + Normalize
    ↓
Pretrained ResNet18
    ↓
512-D Feature Embedding
    ↓
Compare with Normal Feature Bank
    ↓
Nearest-Neighbor Distance
    ↓
Anomaly Score
    ↓
Normal / Defective
```

## Dataset

**MVTec Anomaly Detection Dataset — Bottle Category**

Training data contains only normal bottle images.

The official test set contains:
- Normal bottles
- Broken large defects
- Broken small defects
- Contamination defects

Normal training images were split into training and validation subsets. The validation-normal subset was used to determine the anomaly threshold, while the official test set remained untouched until final evaluation.

## Model Approach

- Pretrained ResNet18 feature extractor
- 512-dimensional image embeddings
- Euclidean distance for feature comparison
- Nearest-neighbor distance used as anomaly score
- Threshold selected using the 99th percentile of validation-normal anomaly scores

A convolutional autoencoder was initially evaluated as a baseline but failed to separate normal and anomalous images effectively, achieving approximately **0.305 ROC-AUC**. The feature-based ResNet18 approach significantly improved performance.

## Results

| Metric | Result |
|---|---:|
| Accuracy | 93% |
| Defect Precision | 98% |
| Defect Recall | 92% |
| Defect F1-score | 95% |
| ROC-AUC | 0.9905 |

### Confusion Matrix

```text
[[19  1]
 [ 5 58]]
```

The final model correctly detected **58 of 63 defective images**, while missing 5 defects.

## Tech Stack

- Python
- PyTorch
- Torchvision
- ResNet18
- NumPy
- Scikit-learn
- Pillow
- MVTec AD

## Project Structure

```text
industrial-anomaly-detection/
│
├── anomaly_detection.ipynb
├── bottle_anomaly_data.pth
├── requirements.txt
└── README.md
```

## Key Learning

This project demonstrates:

- CNN feature extraction
- Transfer learning
- One-class anomaly detection
- Feature embeddings
- Nearest-neighbor anomaly scoring
- Leakage-safe validation and testing
- Threshold selection
- Model evaluation using precision, recall, F1-score and ROC-AUC
- Error analysis and model iteration