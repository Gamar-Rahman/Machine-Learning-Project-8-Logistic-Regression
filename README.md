# Machine-Learning-Project-8-Logistic-Regression
A hands-on machine learning project focused on implementing Logistic Regression from scratch for binary and multi-class classification.

### Project Overview

This lab demonstrates how logistic regression works at both the mathematical and implementation levels. Instead of relying only on built-in models, the notebook builds the core pieces manually, including:
Sigmoid function
Binary logistic loss
Gradient descent training
Softmax function
Cross-entropy loss
Multinomial logistic regression
The project uses the scikit-learn Digits dataset to classify handwritten digits and evaluate model performance on real data.

### Objectives

Understand the foundations of logistic regression
Implement binary classification from scratch
Extend the model to multi-class classification
Train models with gradient descent
Evaluate classification performance on real-world digit data

### Key Concepts

# Binary Logistic Regression
Logistic regression predicts the probability of a binary outcome using the sigmoid function:
[
\sigma(z) = \frac{1}{1 + e^{-z}}
]
This allows the model to output values between 0 and 1, which can be converted into class predictions.

# Logistic Loss
The notebook computes the binary logistic loss and its gradient to optimize model weights with gradient descent.

# Multinomial Logistic Regression
For multi-class classification, the notebook extends logistic regression using:
Softmax
Cross-entropy loss
This enables classification across all 10 digit classes.

### Dataset

This project uses the Optical Recognition of Handwritten Digits dataset from `sklearn.datasets`.
Binary experiment: digits 0 vs 1
Multi-class experiment: digits 0 through 9
Each image is an 8x8 grayscale image
Each sample is represented by 64 features

### Workflow

Load the digits dataset
Visualize sample images
Pad features where needed
Split into train/test sets
Normalize features with `StandardScaler`
Train binary logistic regression with gradient descent
Evaluate testing accuracy
Train multinomial logistic regression
Evaluate multi-class testing accuracy
Plot training loss

### Results

# Binary Classification
Task: Distinguish digit 0 from digit 1
Testing Accuracy: 97.22%

# Multi-Class Classification
Task: Classify digits 0–9
Testing Accuracy: 97.78%
These results show that even a relatively simple linear classifier can perform strongly when the data is clean and well-preprocessed.

### Security Relevance
Although this lab is centered on handwritten digit recognition, the same logistic regression concepts are highly relevant in cybersecurity and security analytics.
Examples include:
phishing email classification
malicious vs benign traffic detection
user behavior anomaly detection
fraud detection
binary alert triage workflows
multi-class attack categorization
This project reinforces a practical lesson: simple, interpretable models can still provide strong baseline performance for security use cases.

### Repository Structure
```bash
lab-8-logistic-regression/
├── Lab_8_Solution.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── linkedin_post.md
```
### Installation
```bash
git clone https://github.com/your-username/lab-8-logistic-regression.git
cd lab-8-logistic-regression
pip install -r requirements.txt
```
### Requirements

Python 3.10+
NumPy
Matplotlib
scikit-learn
Jupyter

