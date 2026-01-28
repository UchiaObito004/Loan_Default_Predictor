# Loan Default Predictor

## Overview

This project builds an end-to-end machine learning pipeline to predict whether a loan applicant is likely to default. The focus is on creating a clean, reliable, and reproducible workflow rather than just training a model.

The final model achieves around **97.6% accuracy** with a **ROC–AUC score close to 1.0**, indicating strong predictive performance.

---

## Problem Statement

Loan defaults are a major risk for financial institutions. Manual evaluation is slow and inconsistent. This project uses machine learning to automate risk assessment and support data-driven loan approval decisions.

---

## Workflow

1. **Data Ingestion & EDA**

   * Analyzed feature distributions, missing values, skewness, and outliers
   * Used EDA insights to guide preprocessing decisions

2. **Data Preprocessing & Feature Engineering**

   * Skewness correction using **Yeo–Johnson transformation**
   * Outlier handling using the **IQR method**
   * Categorical feature encoding
   * Numerical feature scaling

3. **Model Training**

   * Trained a **Random Forest Classifier** as the base model
   * Chosen for robustness and strong performance on tabular data

4. **Hyperparameter Optimization**

   * Used **Optuna** to tune key hyperparameters
   * Improved generalization and overall performance

5. **Model Evaluation**

   * Accuracy: ~97.6%
   * ROC–AUC score close to 1.0

6. **Deployment**

   * Deployed the full pipeline using **Streamlit** for interactive predictions

7. **Continuous Integration**

   * Added CI to ensure code quality, reproducibility, and reliable updates

---

## Tech Stack

* Python
* Pandas, NumPy, Scikit-learn
* Optuna
* Streamlit
* GitHub Actions (CI)

---

## Key Takeaways

* Complete end-to-end ML pipeline
* Strong preprocessing and feature engineering
* Hyperparameter optimization with Optuna
* Deployment-ready with CI integration
