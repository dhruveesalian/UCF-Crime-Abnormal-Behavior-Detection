# UCF-Crime Abnormal Behavior Detection

## Overview

This project implements a deep-learning baseline for detecting abnormal behaviour in public-space surveillance imagery using the UCF-Crime dataset.

The implementation is based on the approaches reviewed in the paper:

**“Detecting Abnormal Behavior Events and Gatherings in Public Spaces Using Deep Learning: A Review”**

The review discusses several deep-learning approaches for abnormal-event detection, including CNNs, autoencoders, 3D-VAE methods, and transfer learning.

## Dataset

**Dataset:** UCF-Crime

The Kaggle version used in this project contains extracted image frames from UCF-Crime videos.

The dataset contains 14 classes, including:

* Abuse
* Arrest
* Arson
* Assault
* Burglary
* Explosion
* Fighting
* NormalVideos
* RoadAccidents
* Robbery
* Shooting
* Shoplifting
* Stealing
* Vandalism

For this implementation, the task was converted into a binary classification problem:

* **Normal**
* **Abnormal**

A balanced subset was used for the experiment:

* Training: 1,000 normal + 1,000 abnormal images
* Testing: 300 normal + 300 abnormal images
* Random seed: 42

## Model

**Architecture:** ResNet18 with transfer learning

The pretrained ResNet18 model was adapted for binary classification by replacing its final classification layer with a single-output layer.

### Training configuration

* Epochs: 5
* Batch size: 32
* Input size: 224 × 224
* Optimizer: Adam
* Device: NVIDIA Tesla T4 GPU
* Image normalization: ImageNet normalization

**Important:** The review paper does not specify one universal architecture or one universal number of training epochs because it is a review of multiple studies. Therefore, the 5-epoch ResNet18 configuration is an implementation choice for this project, not a value taken directly from the review.

## Results

Results on the 600-image balanced test subset:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 73.17% |
| Precision | 70.62% |
| Recall    | 79.33% |
| F1 Score  | 74.73% |
| ROC-AUC   | 0.8151 |

### Confusion Matrix

The confusion matrix is available in:

`results/UCF_Crime_Confusion_Matrix.png`

### ROC Curve

The ROC curve is available in:

`results/UCF_Crime_ROC_Curve.png`

### Results Data

The numerical results are available in:

`results/UCF_Crime_Results.csv`

## Project Structure

```text
UCF-Crime-Abnormal-Behavior-Detection/
│
├── results/
│   ├── UCF_Crime_Confusion_Matrix.png
│   ├── UCF_Crime_ROC_Curve.png
│   └── UCF_Crime_Results.csv
│
├── model/
│   └── resnet18_ucf_crime_baseline.pth
│
├── notebook/
│   └── UCF_Crime_DL.ipynb
│
└── README.md
```

## Reference

Rodrigo-Guillen, R., Garcia-D’Urso, N., Mora-Mora, H., & Azorin-Lopez, J. (2025).

**Detecting Abnormal Behavior Events and Gatherings in Public Spaces Using Deep Learning: A Review.**

Journal of Sensor and Actuator Networks, 14(4), 69.

DOI: 10.3390/jsan14040069
