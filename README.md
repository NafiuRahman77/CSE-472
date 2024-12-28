# CSE 472: Machine Learning Sessional

This repository contains the assignments and solutions for the **CSE 472: Machine Learning Sessional** course. Each offline session focuses on key machine learning concepts and algorithms, with hands-on implementations and experimentation.

---

## Contents

1. [Offline 1: Data Preprocessing and Feature Engineering](#Offline-1)
2. [Offline 2: Ensemble Learning with Logistic Regression](#Offline-2)
3. [Offline 3: Neural Network and Backpropagation](#Offline-3)
4. [Offline 4: PCA and Expectation-Maximization](#Offline-4)

---

## Offline 1: Data Preprocessing and Feature Engineering

### **File**: `1905077.ipynb`

### **Introduction**
This assignment covers data preprocessing and feature engineering for machine learning models. It includes tasks like cleaning raw data, handling missing values, normalizing datasets, and feature selection.

### **Key Tasks**
- Import and preprocess the "IBM HR Analytics Employee Attrition & Performance" dataset.
- Handle missing values, redundancy, and data normalization.
- Convert categorical variables into numerical representations.
- Perform correlation analysis to identify important features.
- Prepare the dataset for a machine learning pipeline.

### **Dataset**
- **Source**: [IBM HR Analytics Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## Offline 2: Ensemble Learning with Logistic Regression

### **File**: `1905077.ipynb` (Datasets in the `data/` folder)

### **Introduction**
This assignment focuses on implementing Logistic Regression (LR) from sratch, ensemble learning techniques using bagging and stacking and LR the base classifier.

### **Key Tasks**
- Preprocess datasets to standardize input formats.
- Implement Logistic Regression (LR) as the base learner.
- Implement Bagging with 9 LR models and Stacking with LR as the meta-classifier.
- Create a simple majority voting-based ensemble for comparison.
- Evaluate model performance using metrics and violin plots.

### **Datasets**
1. [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
2. [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
3. [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## Offline 3: Neural Network and Backpropagation

### **File**: `1905066.ipynb`

### **Introduction**
This assignment involves implementing a Feed-Forward Neural Network (FNN) from scratch for apparel classification.

### **Key Components**
- **Dense Layer**: Fully connected layer.
- **Batch Normalization**: Normalizes the input for each layer.
- **ReLU Activation**: Activation function for hidden layers.
- **Dropout**: Regularization to prevent overfitting.
- **Adam Optimizer**: Adaptive moment estimation for weight updates.
- **Softmax Regression**: For multi-class classification.

### **Key Tasks**
- Modularize the implementation to allow flexibility in architecture.
- Implement backpropagation and mini-batch gradient descent for training.
- Train and evaluate the FNN using the provided dataset.

---

## Offline 4: PCA and Expectation-Maximization

### **Files**: `em.ipynb` and `pca.ipynb`

### **Part 1: Principal Component Analysis (PCA)**

#### **Dataset**: `pca_data.txt`
- Contains 1000 rows and 500 columns representing 1000 sample points with 500 features each.

#### **Key Tasks**
1. Perform PCA for dimensionality reduction.
2. Project data along the two eigenvectors corresponding to the highest eigenvalues.
3. Create a 2D scatter plot for visualization.
4. Generate UMAP and t-SNE plots using library functions for comparison.

---

### **Part 2: Expectation-Maximization (EM)**

#### **Dataset**: `em_data.txt`
- Represents the number of children in 1000 families, with some given family planning advice.

#### **Key Tasks**
1. Implement the EM algorithm for Poisson mixture models.
2. Estimate:
   - Mean number of children in families with and without family planning.
   - Proportion of families with and without family planning.

---

## How to Use

### Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>

### Run the relevant .ipynb files under the folders
