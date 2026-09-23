## Results

The final ResNet18 model was trained using a balanced subset of the UCF-Crime extracted-frame dataset.

### Dataset Split

- Training images: 20,000
  - Normal: 10,000
  - Abnormal: 10,000
- Testing images: 4,000
  - Normal: 2,000
  - Abnormal: 2,000

### Performance

| Metric | Score |
|---|---:|
| Accuracy | 75.28% |
| Precision | 71.18% |
| Recall | 84.95% |
| F1-Score | 77.46% |
| ROC-AUC | 0.7927 |

### Model

- Architecture: ResNet18
- Approach: Transfer Learning
- Fine-tuning: Final ResNet18 convolutional block + classifier
- Loss function: Binary Cross Entropy with Logits
- Optimizer: Adam
- Input size: 224 × 224
- Classification: Normal / Abnormal

The final trained model is available in the [v2.0 GitHub Release](../../releases/tag/v2.0).
