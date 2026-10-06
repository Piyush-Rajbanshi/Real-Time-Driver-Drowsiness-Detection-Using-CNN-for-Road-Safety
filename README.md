# Real-Time Driver Drowsiness Detection Using CNN for Road Safety

## TECH 405 — Week 6/7/8 Project

### Classification task
Binary eye-state classification:
- Closed Eyes = 0
- Open Eyes = 1

The project uses the MRL eye-state dataset and compares two distinct models:
1. MLP neural network
2. Tuned CNN deep-learning model (Week 4 Experiment 3)

### Dataset
- 4,000 grayscale eye images
- 2,000 Closed Eyes
- 2,000 Open Eyes
- 16 subjects
- Subject-level split
  - Train: 2,127
  - Validation: 236
  - Test: 1,637
- Input size: 64 x 64
- Pixel normalization: 0–1
- Training-only augmentation: horizontal flip, small rotation, small zoom

Dataset source:
https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset

### Current verified test results
Binary precision/recall/F1 use Open Eyes = 1 as the positive class.

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| MLP Neural Network | 97.50% | 99.71% | 96.40% | 98.03% | 0.9811 |
| CNN Deep Learning | 97.25% | 99.71% | 96.02% | 97.83% | 0.9928 |

Safety-focused Closed Eyes metrics:
- MLP: Precision 93.83%, Recall 99.48%, F1 96.57%
- CNN: Precision 93.23%, Recall 99.48%, F1 96.25%

Confusion matrices (rows = actual Closed/Open, columns = predicted Closed/Open):
- MLP: [[578, 3], [38, 1018]]
- CNN: [[578, 3], [42, 1014]]

### Interpretation
The MLP is slightly better on accuracy, Open-Eyes recall, and F1 at the 0.50 threshold. The CNN has the higher ROC-AUC, indicating stronger threshold-independent score separation. Both models have 99.48% recall for Closed Eyes.

This project is an eye-state classifier and should not be described as a clinical or safety-certified drowsiness detector. Real deployment would require temporal context, broader real-world validation, calibrated thresholds, and additional cues such as blink duration, head pose, gaze, and yawning.

### Files
- `Week8_Driver_Drowsiness_Classification_Presentation.pptx` — final presentation
- `Week6_7_Driver_Drowsiness_Full_Code.py` — notebook-derived full code
- `Week8Assignment.ipynb` — original Colab notebook used for the project
