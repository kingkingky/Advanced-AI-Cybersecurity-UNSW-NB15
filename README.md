# Advanced AI for Cybersecurity: Intrusion Detection with UNSW-NB15

## Overview

This project develops a machine learning-based Intrusion Detection System (IDS) using the UNSW-NB15 dataset.

The objective is to classify network traffic into multiple attack categories and distinguish malicious activities from normal behavior.

## Dataset

UNSW-NB15 Dataset

Features:
- Network traffic statistics
- Protocol information
- Service information
- State information

Target:
- attack_cat

## Models Compared

- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- K-Nearest Neighbors

## Best Model

Random Forest

Accuracy: 82.28%

Weighted F1 Score: 80.14%

## Feature Engineering

Additional features:

- bytes_ratio
- packet_ratio
- load_ratio
- pkt_size_ratio

## Results

### Model Comparison

![Model Comparison](images/model_comparison.png)

### Feature Importance

![Feature Importance](images/feature_importance.png)

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

## Technologies

- Python
- Pandas
- Scikit-Learn
- Matplotlib

## Author

Kankrit