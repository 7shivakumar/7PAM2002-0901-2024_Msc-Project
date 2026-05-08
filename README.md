# Predicting Loan Approval Status Using Machine Learning

**MSc Data Science Dissertation Project | University of Hertfordshire | Distinction**

A comparative machine learning study that predicts loan approval decisions using Neural Networks, Gradient Boosting, and Random Forest — with a deployed FastAPI web application for real-time predictions.

---

## Project Overview

Loan approval decisions have traditionally relied on manual evaluation by loan officers, which is time-consuming, inconsistent, and prone to human bias. This project builds and compares three machine learning models to automate loan approval predictions based on customer financial and demographic data.

**Research Question:**
> How do Neural Networks, Gradient Boosting, and Random Forest compare in predicting loan approval status, and which model provides the highest accuracy for automating loan decisions?

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Neural Network | 96.49% | 95.57% | 94.97% | 95.27% |
| Gradient Boosting | 95.55% | 100.00% | 88.05% | 93.65% |
| **Random Forest** | **97.78%** | **98.07%** | **95.91%** | **96.98%** |

**Random Forest achieved the highest accuracy of 97.78%** — outperforming both Neural Networks and Gradient Boosting across all evaluation metrics.

---

## Dataset

- **Source:** [Kaggle — Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- **License:** MIT
- **Records:** 614 loan applications
- **Features:** 13 (including income, loan amount, credit score, education, employment status, asset values)
- **Target Variable:** Loan Status (Approved / Rejected)

### Key Features

| Feature | Description |
|---|---|
| income_annum | Annual income of the applicant |
| loan_amount | Requested loan amount |
| cibil_score | Credit score of the applicant |
| education | Graduate / Not Graduate |
| self_employed | Employment status |
| loan_term | Duration of the loan |
| residential_assets_value | Value of residential assets |
| commercial_assets_value | Value of commercial assets |

---

## Methodology

### 1. Exploratory Data Analysis
- Analysed distribution of loan approval status, education, and employment
- Identified skewed distributions in income and loan amount
- Investigated missing values across key features

### 2. Data Preprocessing
- **Missing values:** Median imputation for numerical features, mode imputation for categorical features
- **Encoding:** Label encoding for categorical variables (Gender, Education, Employment)
- **Scaling:** StandardScaler applied to normalise numerical features (mean=0, std=1)
- **Outlier handling:** Retained high-income and high-loan outliers to preserve dataset representativeness

### 3. Model Training
Three machine learning models were trained on an 80/20 train-test split:

- **Neural Network (MLP):** 1 hidden layer, 10 neurons, ReLU activation, Adam optimiser
- **Gradient Boosting:** 12 estimators, learning rate 0.057721, max depth 3
- **Random Forest:** 100 decision trees, random state 42

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices for detailed error analysis

### 5. Deployment
Trained models deployed via **FastAPI** — a REST API that accepts loan application data and returns real-time approval predictions.

---

## Key Findings

- **Random Forest** is the most reliable model, achieving the best balance across all metrics with minimal false positives (6) and false negatives (13)
- **Gradient Boosting** achieved perfect precision (100%) but lower recall (88.05%), making it suitable for high-risk scenarios where false approvals must be avoided
- **Neural Networks** performed consistently but require careful hyperparameter tuning to avoid overfitting
- **Credit score, income, and loan amount** were identified as the most influential features in predicting loan approval

---

## Tech Stack

- **Python 3**
- **Scikit-learn** — model training and evaluation
- **Pandas and NumPy** — data manipulation and analysis
- **Matplotlib and Seaborn** — data visualisation
- **FastAPI** — model deployment and REST API
- **Joblib** — model serialisation
- **Jupyter Notebook** — development environment

---

## Project Structure

```
7PAM2002-0901-2024_Msc-Project/
│
├── Untitled.ipynb              # Full Python notebook with code and analysis
├── loan_approval_dataset.csv   # Dataset (MIT licence)
├── 7PAM2002-0901-2024.pdf      # Full MSc dissertation report
└── README.md                   # Project documentation
```

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/7shivakumar/7PAM2002-0901-2024_Msc-Project.git
cd 7PAM2002-0901-2024_Msc-Project
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn fastapi uvicorn joblib
```

3. Open the notebook
```bash
jupyter notebook Untitled.ipynb
```

4. Run all cells to train models and generate results

---

## API Usage

The FastAPI deployment accepts POST requests with loan application data:

```python
POST /predict/
{
  "no_of_dependents": 2,
  "education": 1,
  "self_employed": 0,
  "income_annum": 50000,
  "loan_amount": 200000,
  "loan_term": 12,
  "cibil_score": 700,
  "residential_assets_value": 150000,
  "commercial_assets_value": 0,
  "luxury_assets_value": 10000,
  "bank_asset_value": 20000,
  "model_choice": "rf"
}
```

Response:
```json
{"model": "rf", "result": "Approved"}
```

---

## Academic Context

- **Institution:** University of Hertfordshire, School of Physics, Engineering and Computer Science
- **Programme:** MSc Data Science
- **Module:** 7PAM2002 — Data Science Final Project
- **Supervisor:** Peter Scicluna
- **Result:** Distinction
- **Submitted:** January 2025

---

## Author

**Shiva Kumar**
- Email: sandy.shiva73@gmail.com
- LinkedIn: linkedin.com/in/shivakumar
- GitHub: github.com/7shivakumar
