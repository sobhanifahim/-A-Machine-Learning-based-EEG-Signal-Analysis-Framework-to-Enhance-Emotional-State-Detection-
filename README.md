This repository contains an end-to-end framework for EEG-based emotion recognition, integrating advanced signal preprocessing, feature extraction, synthetic data generation, and machine learning classification. The framework is designed to improve the robustness and accuracy of emotional state detection using 8-channel EEG recordings.

🔑 Key Features

Preprocessing Pipeline

FIR band-pass filtering (order 100) to remove drifts and noise.

Independent Component Analysis (ICA) for ocular and muscular artifact removal.

Feature Extraction

Discrete Wavelet Transform (DWT) to extract energy and Shannon Entropy features.

Sub-band analysis: low/high Alpha, Beta, and Gamma frequency bands.

Synthetic Data Generation

TVAE (Tabular Variational Autoencoder) Synthesizer to address class imbalance.

Achieved 93% data quality, improving model generalization.

Classification

Emotion categories: Positive, Neutral, Depressed, Anxiety.

Supports multiple ML algorithms (Random Forest, SVM, Neural Networks).

5-fold cross-validation for performance evaluation.

Evaluation Metrics

Accuracy, Precision, Recall, F1-score, and Confusion Matrix analysis.
