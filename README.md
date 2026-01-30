# Temporal Behavior Modeling for Sequential Anomaly Detection

## Project Overview
This repository contains an end-to-end deep learning pipeline designed to detect Insider Threats by analyzing sequential user behavior. Leveraging a Long Short-Term Memory (LSTM) architecture, the system processes multi-dimensional behavioral features such as login frequency, file operations, and USB events to identify anomalies that deviate from established user patterns.

The model was trained on a large-scale dataset of 1.3M+ records which was derived from the CERT Insider Threat Dataset, achieving a ROC-AUC of 0.9998 by successfully addressing extreme class imbalance and temporal dependency challenges.

## Key Features
* **User-Aware Sequence Engineering:** Custom data pipeline that groups data by user and time, ensuring 7-day sliding windows never cross-contaminate between different individuals.
* **Imbalance Mitigation:** Utilized Weighted Binary Cross-Entropy (BCE) Loss to handle a 4.4% minority class, ensuring high sensitivity to rare suspicious events.
* **Scalable Architecture:** Optimized LSTM configuration with 128 hidden units and dropout regularization for robust feature extraction.

##  Performance Metrics
The model demonstrates reliability with the following results:

| Metric | Value |
| :--- | :--- |
| **ROC-AUC** | 0.9998 |
| **Precision** | 0.9841 |
| **Recall** | 0.9970 |
| **F1-Score** | 0.9905 |

## Technical Stack
* **Core:** Python, PyTorch
* **Data Engineering:** Pandas, NumPy
* **Modeling & Metrics:** Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

