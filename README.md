#  Heart Disease Prediction using Machine Learning

## Overview

This project predicts whether a patient is likely to have heart disease using a Machine Learning model trained on the UCI Heart Disease dataset.

The project includes:
- Data preprocessing and exploratory data analysis
- Model comparison across multiple classification algorithms
- Feature selection using Random Forest feature importance
- Hyperparameter tuning using RandomizedSearchCV
- An interactive Streamlit web application for prediction

---

## Features

- Predicts the likelihood of heart disease
- Displays prediction confidence
- Interactive and user-friendly web interface
- Input validation with user-friendly error messages
- Feature-selected Random Forest model

---

## Machine Learning Workflow

The following models were evaluated:

- Dummy Classifier (Baseline)
- Logistic Regression
- Decision Tree
- Gradient Boosting
- Random Forest

Random Forest achieved the best overall performance.

Feature selection was then performed using Random Forest feature importance.

Hyperparameter tuning was carried out using RandomizedSearchCV. Although the tuned model produced similar performance, the feature-selected Random Forest achieved a slightly higher F1-score and was therefore selected as the final deployed model.

---

## Dataset

**Dataset:** Heart Disease Dataset (Compiled from the UCI Heart Disease Dataset)

Features used by the final model:

- Age
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Resting ECG
- Maximum Heart Rate
- Oldpeak

Target:

- Heart Disease (0 = No, 1 = Yes)

---

## Technologies Used

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Joblib

- streamlit == 1.60.0
-joblib==1.4.2
-pandas == 3.0.5
-scikit-learn == 1.9.0
-numpy == 2.5.1
-matplotlib

---

## Project Structure

```
.
├── program_code.ipynb
├── streamlit_application.py
├── mldp_project_model.pkl
├── requirements.txt
└── README.md
```

---

## Installation


Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run streamlit_application.py
```

---

## Model Performance

Final model:

- Random Forest (Feature Selected)

Performance:

- Accuracy: 0.874
- Precision: 0.872
- Recall: 0.909
- F1-score: 0.890



---

## Author

Developed as part of a Machine Learning project.
