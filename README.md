[github_readme_ Datheon Internship.md](https://github.com/user-attachments/files/34849367/github_readme_.Datheon.Internship.md)
# Datheon Machine Learning & Data Science Internship

**4-Week Hands-On Project Engineering Portfolio | Supervised Classification, Regularized Regression, Imbalanced Ensemble Modeling & NLP Sentiment Analysis**

* **Program:** Machine Learning & Data Science Internship (Batch 01)
* **Organization:** Datheon
* **Application ID:** `DT-2026-549PF`
* **Repository Name:** `Datheon_Machine_Learning_&_Data_Science_Internship`

---

## Repository Overview

This repository contains end-to-end machine learning and natural language processing solutions developed for the **Datheon 4-Week Machine Learning & Data Science Internship**. Each task is structured as a self-contained module with reproducible preprocessing pipelines, model training, hyperparameter tuning, and visual diagnostics.

---

## Repository Folder Structure

```text
Datheon_Machine_Learning_&_Data_Science_Internship/
├── Task_1_Multi_Class_Classification/
│   └── iris_wine_classification.ipynb
├── Task_2_Housing_Price_Prediction/
│   ├── housing_price_prediction.ipynb
│   └── housing dataset.csv
├── Task_3_Customer_Churn_Classifier/
│   └── customer_churn_prediction.ipynb
├── Task_4_NLP_Sentiment_Classification/
│   └── sentiment_analysis_tfidf.ipynb
└── README.md
```

---

## Program Tasks & Technical Breakdown

### Task 1: Multi-Class Benchmark Classification Pipeline (Iris / Wine)
* **Objective:** Build an end-to-end supervised classification pipeline on the Iris/Wine benchmark dataset to evaluate linear and tree-based classifiers.
* **Technical Approach:**
  1. Load and explore the dataset using `pandas` and pairwise visualizations with `seaborn`.
  2. Standardize continuous features via `StandardScaler()` and split data into 80% training and 20% testing subsets.
  3. Train `LogisticRegression` and `DecisionTreeClassifier` models.
  4. Benchmark performance across **Accuracy**, **Precision**, **Recall**, and **F1-Score**, supported by confusion matrix heatmaps and decision boundary plots.
* **Tools:** `Python`, `Scikit-Learn`, `Pandas`, `Seaborn`, `Matplotlib`

---

### Task 2: Real Estate Housing Price Prediction (Ridge & Lasso Regression)
* **Objective:** Predict continuous California median house values (`median_house_value`) while addressing missing records, skewed distributions, and multicollinearity.
* **Technical Approach:**
  1. Clean currency string formatting (`$`, `,`), strip header whitespace, and impute 207 missing values in `total_bedrooms` using median imputation.
  2. Compute correlation heatmaps and engineer domain interaction features (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`, `income_x_rooms`).
  3. Diagnose multicollinearity using Variance Inflation Factor (VIF) and drop collinear raw count columns.
  4. Standardize continuous predictors (`StandardScaler`) and one-hot encode `ocean_proximity` (`OneHotEncoder`).
  5. Tune **Ridge (L2)** and **Lasso (L1)** models across multiple `alpha` values via 5-fold `GridSearchCV` and compare against **Ordinary Least Squares (OLS)** using **RMSE**, **MAE**, and **$R^2$**.
  6. Generate Predicted vs. Actual scatter plots and Kernel Density Estimate (KDE) residual error curves.
* **Tools:** `Python`, `Scikit-Learn (LinearRegression, Ridge, Lasso, GridSearchCV)`, `Statsmodels`, `Seaborn`

---

### Task 3: Predictive Customer Churn Classifier (Random Forest & XGBoost)
* **Objective:** Predict telecom customer attrition on imbalanced subscription records and achieve an **ROC-AUC score > 80%**.
* **Technical Approach:**
  1. Load the Telco Customer Churn dataset, convert `TotalCharges` to numeric values, and handle missing entries.
  2. Encode categorical features via `OneHotEncoder` and address severe class imbalance using **SMOTE** (`imbalanced-learn`) or class weighting (`scale_pos_weight`).
  3. Train and cross-validate `RandomForestClassifier` and `XGBClassifier` ensemble models.
  4. Plot **ROC-AUC curves**, generate confusion matrices, and extract the top 10 most influential churn features.
* **Tools:** `Python`, `Scikit-Learn`, `XGBoost`, `SMOTE (imbalanced-learn)`, `Matplotlib`

---

### Task 4: NLP Sentiment Classification on Customer Reviews (TF-IDF)
* **Objective:** Construct a natural language processing pipeline to classify positive and negative polarity across customer product or movie reviews.
* **Technical Approach:**
  1. Ingest raw text reviews and clean the corpus using `NLTK` and regular expressions (lowercasing, HTML tag removal, punctuation stripping, stopword filtering, and lemmatization).
  2. Transform preprocessed text into numerical feature vectors using `TfidfVectorizer(max_features=5000)`.
  3. Train and benchmark `MultinomialNB` (Multinomial Naive Bayes) and `LogisticRegression` classifiers.
  4. Implement an interactive `predict_sentiment()` helper function to evaluate custom live user text inputs.
* **Tools:** `Python`, `NLTK / SpaCy`, `TfidfVectorizer`, `Scikit-Learn`

---

## Performance Summary

*(Update the metrics below with your final notebook results)*

| Task | Model(s) | Primary Evaluation Metric | Result |
| :--- | :--- | :--- | :---: |
| **Task 1** | Logistic Regression / Decision Tree | Accuracy / F1-Score | `[INSERT_METRIC]` |
| **Task 2** | OLS / Ridge (L2) / Lasso (L1) | RMSE / MAE / $R^2$ | `[INSERT_METRIC]` |
| **Task 3** | Random Forest / XGBoost | ROC-AUC (> 80%) | `[INSERT_METRIC]` |
| **Task 4** | Multinomial Naive Bayes / Logistic Regression | Accuracy / F1-Score | `[INSERT_METRIC]` |

---

## How to Run Locally or in Google Colab

1. **Clone the repository:**
   ```bash
   git clone https://github.com/[YOUR_GITHUB_USERNAME]/Datheon_Machine_Learning_&_Data_Science_Internship.git
   cd Datheon_Machine_Learning_&_Data_Science_Internship
   ```
2. **Install required dependencies:**
   ```bash
   pip install numpy pandas scikit-learn xgboost imbalanced-learn statsmodels nltk matplotlib seaborn
   ```
3. **Run any task notebook:**
   * Open the `.ipynb` notebook for the desired task in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
   * Upload the corresponding dataset when prompted (e.g., `housing dataset.csv` for Task 2) and execute all cells sequentially.
