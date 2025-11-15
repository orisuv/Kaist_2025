Lending Club Loan Default Prediction

This project analyzes Lending Club loan data and builds machine-learning models to predict whether a loan will become charged-off. The goal is to identify high-risk borrowers before loan approval and improve credit risk management.

Project Overview

Performed exploratory data analysis on key variables such as interest rate, debt-to-income ratio, credit grade, revolving utilization, and income.

Conducted full preprocessing: missing value handling, outlier removal, categorical encoding, scaling, and date feature extraction.

Addressed class imbalance using SMOTE.

Engineered additional features such as:

interest_amount = loan_amnt × int_rate

installment_income_ratio = installment / annual_inc

open_to_total_ratio

public_issue_score = pub_rec + pub_rec_bankruptcies

Trained and compared multiple models including Logistic Regression, Gradient Boosting, XGBoost, LightGBM.

Requirements

Packages include:

numpy

pandas

seaborn

matplotlib

scikit-learn

xgboost

lightgbm

statsmodels

imbalanced-learn
