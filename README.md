# Facial Classification through Deep Learning

CS 405 – Deep Learning – Spring 2025  
National University of Sciences and Technology (NUST)  
School of Electrical Engineering and Computer Science (SEECS)

## 🧠 Overview

This project is part of Assignment 2 for the CS405 Deep Learning course. The objective is to design and train deep learning models for **facial classification** using CNN architectures and transfer learning. The project also includes performance evaluation through a Kaggle competition.

## 📂 Provided Materials

- Training data (`train/`): 140,000+ facial images across 7000 identities.
- Testing data (`test/`): 35,000 unlabeled images.
- `sample_submission.csv`: Submission format for Kaggle.
- Dataset and competition hosted on:  
  [Kaggle Competition Link](https://www.kaggle.com/competitions/face-classification-deep-learning-cs-405/overview)

## 🚀 Features

- Custom train/validation split from training data.
- Deep CNN models using transfer learning.
- Data augmentation and preprocessing.
- Performance monitoring with training/validation accuracy/loss plots.
- CSV submission for Kaggle leaderboard evaluation.
- Modular and well-documented code.

## 🛠️ Technologies Used

- Python 3.x
- PyTorch
- NumPy, Pandas, Matplotlib
- Jupyter Notebook
- Kaggle API 

## 🧪 Model Architecture

- Utilized at least **two CNN backbones** (e.g., ResNet, EfficientNet).
- Transfer learning with customized classification heads.
- Techniques:
  - Dropout
  - Batch Normalization
  - Data Augmentation (Flip, Rotation, Normalize)
  - Learning rate scheduling
- Optimizers: Adam, SGD (experimented)
- Loss: CrossEntropyLoss

## 📊 Training & Evaluation

- Plotted training vs. validation loss and accuracy curves.
- Trained with different hyperparameters to optimize performance.
- Final model used to predict test set labels.
- Submission generated as `submission.csv`.

## 📝 Deliverables

- `notebooks/`: All Jupyter Notebooks used for model training and evaluation.
- `submission.csv`: Kaggle-compatible CSV file with predictions.
- `report.pdf`: Discussion and analysis of results.
- `README.md`: Project documentation.

## 🧾 Report Highlights

- Summary of dataset handling, architecture design, and training.
- Comparison between different CNNs.
- Discussion on:
  - Best hyperparameter settings
  - Real-world applications
  - Model performance insights 
