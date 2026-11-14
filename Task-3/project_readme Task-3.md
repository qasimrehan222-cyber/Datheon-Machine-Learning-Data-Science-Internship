# Predictive Customer Churn Classifier (Random Forest & XGBoost)

An end-to-end machine learning pipeline built in Google Colab to predict customer attrition using telecommunications subscription records. The workflow addresses severe class imbalance using SMOTE and optimizes ensemble classifiers (`RandomForestClassifier` and `XGBClassifier`) to achieve a **>80% ROC-AUC** score.

---

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset Overview](#dataset-overview)
- [Technical Approach & Pipeline](#technical-approach--pipeline)
- [Technologies Used](#technologies-used)
- [How to Run in Google Colab](#how-to-run-in-google-colab)
- [Expected Deliverables & Results](#expected-deliverables--results)

---

## 🎯 Problem Statement
Customer churn is a critical revenue driver in the telecommunications industry. Because the vast majority of customers remain subscribed, churn datasets exhibit significant class imbalance (~73.5% retained vs. ~26.5% churned). Standard classifiers trained on imbalanced data tend to favor the majority class and fail to identify at-risk customers.

**Objective:** Build an ensemble machine learning pipeline that preprocesses mixed data types, handles class imbalance without data leakage, and distinguishes between churning and non-churning customers with an **ROC-AUC score exceeding 80%**.

---

## 📊 Dataset Overview
- **Source File:** `Telco-Customer-Churn dataset.csv`
- **Records:** 7,043 customer accounts
- **Target Variable:** `Churn` (`Yes` = 1, `No` = 0)
- **Key Features:**
  - **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - **Account Information:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
  - **Subscribed Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

---

## ⚙️ Technical Approach & Pipeline

1. **Data Cleaning & Imputation:**
   - Removed non-predictive `customerID` column.
   - Converted `TotalCharges` from string format to numeric (`float64`), handling blank entries for brand-new customers (`tenure == 0`) by imputing `0.0`.
   - Binary-encoded the target variable `Churn`.
2. **Leakage-Free Preprocessing & Resampling (`imblearn.pipeline.Pipeline`):**
   - **Stratified Train/Test Split:** 80% training / 20% testing split preserving the original class distribution.
   - **Feature Transformation (`ColumnTransformer`):** Applied `StandardScaler` to numerical features and `OneHotEncoder(drop='if_binary', handle_unknown='ignore')` to categorical features.
   - **Class Balancing (`SMOTE`):** Applied Synthetic Minority Over-sampling Technique exclusively inside cross-validation and training folds to balance target classes without leaking validation information.
3. **Model Training & Cross-Validation:**
   - Evaluated both **Random Forest** and **XGBoost** classifiers using **5-Fold Stratified Cross-Validation** scored on `roc_auc`.
   - Trained final pipelines on the full training partition and evaluated against the unseen test set.

---

## 🛠️ Technologies Used
- **Environment:** Google Colab / Jupyter Notebook
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning & Preprocessing:** `scikit-learn`, `xgboost`
- **Imbalanced Learning:** `imbalanced-learn` (`SMOTE`)
- **Visualization:** `matplotlib`, `seaborn`

---

## 🚀 How to Run in Google Colab

1. Open a new notebook in [Google Colab](https://colab.research.google.com/).
2. Copy and paste the sequential code cells into your notebook:
   - **Cell 1:** Install (`imbalanced-learn`, `xgboost`) and import required libraries.
   - **Cell 2:** Run `files.upload()` and select `Telco-Customer-Churn dataset.csv` from your local machine.
   - **Cell 3:** Execute data cleaning, `TotalCharges` type conversion, and target encoding.
   - **Cell 4:** Initialize the `ColumnTransformer`, `SMOTE`, and model pipelines.
   - **Cell 5:** Run 5-fold Stratified Cross-Validation and print test set ROC-AUC scores and classification reports.
   - **Cell 6:** Generate and display all evaluation plots.
3. Select **Runtime > Run all** from the top menu to execute the complete pipeline from top to bottom.

---

## 📈 Expected Deliverables & Results

| Metric / Deliverable | Benchmark / Output |
| :--- | :--- |
| **Random Forest Test ROC-AUC** | **~84.3%** (Exceeds >80% target) |
| **XGBoost Test ROC-AUC** | **~84.7%** (Exceeds >80% target) |
| **ROC-AUC Curve** | Comparative plot of Random Forest vs. XGBoost against random guessing baseline |
| **Confusion Matrices** | Side-by-side breakdown of True Positives, False Positives, True Negatives, and False Negatives |
| **Feature Importance Chart** | Horizontal bar chart highlighting the **Top 10 most influential features** driving customer churn (e.g., `Contract_Month-to-month`, `InternetService_Fiber optic`, `tenure`) |