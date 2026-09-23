# UCF-Crime Abnormal Behavior Detection

A deep learning project for detecting abnormal behavior in surveillance-style images using the UCF-Crime dataset and a ResNet18 model.

## 📌 Project Overview

Abnormal behavior detection is an important task in intelligent surveillance systems. The goal of this project is to classify surveillance frames into two categories:

- **Normal**
- **Abnormal**

The project uses **transfer learning with ResNet18** to learn visual patterns associated with abnormal events.

The implementation is based on extracted frames from the **UCF-Crime dataset**.

---

## 🎯 Objectives

The main objectives of this project are:

- Detect abnormal behavior from surveillance images.
- Classify frames as Normal or Abnormal.
- Apply transfer learning using a pretrained ResNet18 network.
- Evaluate the model using standard classification metrics.
- Visualize model performance using a confusion matrix and ROC curve.
- Provide a trained model that can be used for prediction.

---

## 📚 Reference Paper

This project is related to the review paper:

**"Detecting Abnormal Behavior Events and Gatherings in Public Spaces Using Deep Learning: A Review"**

Authors:

- Rafael Rodrigo-Guillen
- Nahuel Garcia-D’Urso
- Higinio Mora-Mora
- Jorge Azorin-Lopez

Journal of Sensor and Actuator Networks, 2025, 14, 69.

DOI:

https://doi.org/10.3390/jsan14040069

The paper reviews deep learning approaches for abnormal behavior and event detection, including CNNs, autoencoders, transfer learning, and video-based architectures.

> **Note:** This project is an independent implementation using the UCF-Crime extracted-frame dataset and ResNet18. The reported results below are from this implementation, not results reported by the review paper.

---

## 📂 Dataset

The project uses the **UCF-Crime dataset**.

The dataset contains surveillance videos representing different types of normal and abnormal activities.

For this implementation, the Kaggle version of the dataset contains **extracted image frames**.

### Classes

The dataset contains 14 classes:

1. Abuse
2. Arrest
3. Arson
4. Assault
5. Burglary
6. Explosion
7. Fighting
8. NormalVideos
9. RoadAccidents
10. Robbery
11. Shooting
12. Shoplifting
13. Stealing
14. Vandalism

For the binary classification experiment, the classes were grouped into:

- **Normal**
- **Abnormal**

---

## 🗂️ Dataset Structure

```text
UCF-Crime
│
├── Train
│   ├── Abuse
│   ├── Arrest
│   ├── Arson
│   ├── Assault
│   ├── Burglary
│   ├── Explosion
│   ├── Fighting
│   ├── NormalVideos
│   ├── RoadAccidents
│   ├── Robbery
│   ├── Shooting
│   ├── Shoplifting
│   ├── Stealing
│   └── Vandalism
│
└── Test
    ├── Abuse
    ├── Arrest
    ├── Arson
    ├── Assault
    ├── Burglary
    ├── Explosion
    ├── Fighting
    ├── NormalVideos
    ├── RoadAccidents
    ├── Robbery
    ├── Shooting
    ├── Shoplifting
    ├── Stealing
    └── Vandalism
