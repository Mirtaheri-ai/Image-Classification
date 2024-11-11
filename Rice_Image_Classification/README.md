# Rice Classification Using Deep Learning

This repository contains an automated system for classifying rice types using Convolutional Neural Networks (CNNs) in TensorFlow. This project focuses on accurately identifying rice varieties based on image and feature data, which is useful for agricultural research, crop management, and quality control in the rice industry.

## Project Overview
This project analyzes and classifies five rice varieties commonly cultivated in Turkey:
- **Arborio**
- **Basmati**
- **Ipsala**
- **Jasmine**
- **Karacadag**

The dataset consists of **75,000 rice grain images** with **15,000 images per variety**. Additionally, a secondary dataset was created with **106 extracted features**:
- **12 morphological features**
- **4 shape-related features**
- **90 color-related features**

## Methodology

### Image Classification
Using **Convolutional Neural Networks (CNNs)**, the project applies various architectures to the image dataset, enabling direct visual classification of rice types.

### Feature Classification
For the extracted features, we employ **Artificial Neural Networks (ANNs)** and **Deep Neural Networks (DNNs)** to classify rice types based on detailed attributes, offering a different perspective beyond raw image data.

## Evaluation Metrics
To ensure model performance and accuracy, several statistical metrics are calculated based on the confusion matrix:
- **Sensitivity and Specificity**
- **Prediction Accuracy and F1 Score**
- **Overall Accuracy**
- **False Positive Rate (FPR) and False Negative Rate (FNR)**

## Results
Each model’s performance is presented in detailed tables, allowing a comprehensive comparison of CNN, ANN, and DNN approaches across image and feature datasets.

## Applications
This rice classification system has practical applications in:
- **Agricultural Research**: Assisting in genetic and crop analysis.
- **Crop Management**: Offering variety-specific insights for care and optimization.
- **Quality Control**: Streamlining identification and categorization processes in the rice production industry.

## Setup and Usage

To start using this project:
1. **Clone the repository**:
   ```bash
   git clone <https://github.com/Mirtaheri-ai/Image-Classification.git>
