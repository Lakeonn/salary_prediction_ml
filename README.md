# Salary Prediction ML Project

A machine learning project that predicts employee salaries based on education, job role, and other relevant features. This repository includes the full workflow: data preprocessing, model training, evaluation, and a prediction interface.

---

## Project Overview

This project demonstrates an end‑to‑end ML pipeline:

- Data ingestion and cleaning  
- Exploratory data analysis (EDA)  
- Feature engineering  
- Model training and evaluation  
- Saving and loading trained models  
- A prediction script for real‑time inference  
- A reproducible environment using Conda  

The goal is to provide a clean, production‑ready structure suitable for consulting, portfolio work, or deployment.

---

## Features

- Clean, modular ML codebase  
- Reproducible environment  
- Logging for auditability  
- Easy‑to‑run training and prediction scripts  
- Notebook‑based experimentation  
- Ready for deployment or extension  

---

## Installation

### 1. Clone the repository

git clone https://github.com/Lakeonn/salary_prediction_ml.git
cd salary_prediction_ml

### 2. Create the Conda environment
conda env create -f environment.yml
conda activate salenv

pip install -r requirements.txt

### 3. Training the Mode
python train.py

### 4. Making Predictions
python predict.py

---

## Data
The dataset includes features such as:
- Age
- Education level
- Job title
- Marital status
- Country
You can replace the dataset with your own as long as the schema matches.

---

## Model Artifacts
Model files are not stored in GitHub due to size limits.
You can generate them by running train.py

---

## Project Motivation

Predicting whether an individual earns more than \$50K per year is a classic and practical classification problem with applications in workforce analytics, compensation strategy, and HR decision‑making. Organizations can use such models to understand the factors most associated with higher income, while job seekers can gain insight into how experience, education, and job roles influence earning potential. This project builds a reproducible machine learning pipeline that classifies income levels using structured demographic and employment data.

---

## Model Details

This project frames salary prediction as a **binary classification task**, where the target variable indicates whether an individual's income is:

- **`>50K`**
- **`<=50K`**

Key aspects of the modeling approach include:

- **Algorithm used:** Random Forest Classifier  
- **Why Random Forest:** Strong performance on tabular data, robustness to noise, ability to capture nonlinear relationships, and built‑in feature importance  
- **Feature engineering:** Encoding categorical variables, handling missing values, scaling or normalizing numerical features when appropriate  
- **Hyperparameters:** Tuned to balance precision and recall, with emphasis on improving F1 score  
- **Model persistence:** The trained classifier is saved locally for inference through the prediction script  

This setup provides a reliable and interpretable baseline for income classification tasks.

---

## Key Modeling Decisions

- Framed the problem as binary classification to align with common compensation thresholds
- Selected F1 score due to class imbalance
- Chose Random Forest for robustness and interpretability on tabular data
- Emphasized reproducibility and auditability over maximum model complexity

---

## Results & Evaluation

Model performance is evaluated using standard classification metrics, with a focus on F1 score due to class imbalance.

### Baseline Comparison

A baseline Logistic Regression model was trained to establish a performance benchmark.
The Random Forest classifier provided improved F1 performance and better recall for the >50K class, making it more suitable for identifying higher-income individuals while controlling false positives.

Key results:
- **F1 Score:** 0.7144  
- **Precision & Recall:** Balanced to reduce false positives and false negatives  
- **Confusion Matrix:** Used to understand classification behavior across both income classes 

These results indicate that the Random Forest model captures meaningful patterns in the data and performs reliably on the binary income classification task.

---

## Limitations

While the model performs well, several limitations should be considered:

- Income data may contain **biases** related to geography, gender, race, or industry 
- Performance depends heavily on the **quality and representativeness** of the dataset  
- The binary threshold (>50K) simplifies income into two categories, which may obscure nuance  

These limitations highlight the importance of careful interpretation and ongoing refinement.

---

## Future Improvements

Potential enhancements include:

- Testing additional models such as XGBoost, LightGBM, or logistic regression with calibrated probabilities  
- Adding explainability tools (e.g., SHAP values) to interpret feature contributions  
- Improving class balance handling through SMOTE or class‑weighted training  
- Adding automated tests and CI/CD workflows  
- Expanding the dataset with more granular features such as job level, certifications, or industry‑specific attributes  

These improvements would strengthen the model’s accuracy, transparency, and usability.

---

## Decision Support Interface (Streamlit)

This project includes an interactive Streamlit application located in the `app/` directory (`stream_app.py`). The interface allows users to input feature values and receive a real‑time prediction of whether an individual's income is likely to be **>50K** or **<=50K**.

The Streamlit app demonstrates how the trained model could be exposed as a lightweight decision support tool for analysts or HR partners, rather than as an autonomous decision-maker.

### Running the Streamlit App

Activate your environment and run:

streamlit run app/stream_app.py

---

## Author

**Olalekan Eniayewu**  
