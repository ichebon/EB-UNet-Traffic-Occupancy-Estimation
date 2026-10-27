# EB-UNet: Enhanced Bottleneck-UNet for Dense Semantic Segmentation & Traffic Occupancy Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

Official implementation of the paper: **"Enhanced Bottleneck-UNet (EB-UNet): Hybrid CNN-SSM Architecture for Privacy-Preserving Traffic Occupancy Estimation"**.

This repository provides a complete framework for dense semantic segmentation on urban driving datasets (**CamVid** and **BDD100K**), featuring a 6-stage encoder-decoder architecture powered by a hybrid **SSM2D + VMamba** bottleneck, along with a mathematically rigorous spatial post-processing pipeline for traffic occupancy monitoring.

---

## 📦 Repository Structure

```text
BU_Unet/
│
├── src/
│   ├── dataset.py        # CamVid & BDD100K data loaders and preprocessing
│   ├── models.py         # EB-UNet (6-stage), SSM2DBottleneck, and baselines
│   ├── metrics.py        # Pixel Accuracy (PA), MAE, RMSE, Pearson Correlation
│   ├── post_process.py   # Morphological filtering (binary dilation N=5, K=3x3)
│   ├── train.py          # Training loop from scratch
│   └── evaluate.py       # Evaluation and occupancy estimation scripts
│
├── scripts/              # Bash/Python runners for training and evaluation
├── config.yaml           # Centralized hyperparameters configuration
├── outputs/              # Saved model weights, logs, and visual predictions
└── README.md
