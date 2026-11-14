# Real Estate Housing Price Prediction (Ridge & Lasso Regression)

**End-to-End Machine Learning Regression Pipeline | Feature Engineering, Multicollinearity Diagnostics & Residual Analysis**

An end-to-end machine learning regression pipeline predicting California median house values using **Ordinary Least Squares (OLS)**, **Ridge (L2)**, and **Lasso (L1)** regularized regression models. Built in Python and Google Colab with domain feature engineering, Variance Inflation Factor (VIF) multicollinearity diagnostics, hyperparameter tuning via `GridSearchCV`, and residual error distribution analysis.

---

## 1. Project Overview & Key Objectives

Predicting real estate prices requires handling skewed distributions, missing records, categorical proximity attributes, and collinear structural features. This repository implements a reproducible preprocessing and modeling workflow to compare unregularized linear regression against L1 and L2 regularized models:

* **Data Cleaning & Imputation:** Normalize column header whitespace, strip currency symbols (`$`, commas) from `median_house_value`, convert target strings to float, and impute 207 missing values in `total_bedrooms` using median imputation.
* **Feature Engineering:** Engineer household-normalized density and interaction features (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`, and `income_x_rooms`).
* **Multicollinearity Mitigation:** Evaluate Variance Inflation Factor (VIF) scores across predictors and drop collinear raw count columns (`total_rooms`, `total_bedrooms`, `population`).
* **Hyperparameter Optimization:** Tune `alpha` regularization penalties across Ridge and Lasso models using 5-fold cross-validation (`GridSearchCV`).
* **Model Evaluation & Diagnostics:** Compare models using **RMSE**, **MAE**, and **$R^2$** scores, supported by Predicted vs. Actual scatter plots and Kernel Density Estimate (KDE) residual error curves.

---

## 2. Dataset Description

* **Source File:** California Housing Dataset (`housing dataset.csv` — 20,640 entries, 10 attributes).
* **Target Variable:** `median_house_value` (continuous USD median house price).
* **Location & Demographics:** `longitude`, `latitude`, `housing_median_age`, `population`, `households`, `median_income`.
* **Property Attributes:** `total_rooms`, `total_bedrooms` (207 missing records imputed via median).
* **Categorical Attribute:** `ocean_proximity` (encoded using `OneHotEncoder`).

---

## 3. Tech Stack & Libraries

* **Environment:** Google Colab / Jupyter Notebook (`.ipynb`)
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (`LinearRegression`, `Ridge`, `Lasso`, `GridSearchCV`, `ColumnTransformer`, `Pipeline`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`)
* **Statistical Diagnostics:** `statsmodels` (`variance_inflation_factor`)
* **Visualization:** `matplotlib`, `seaborn`

---

## 4. Methodology & Workflow

1. **Data Ingestion & Cleaning:** Loaded raw CSV records, stripped leading/trailing whitespace from column headers, cleaned currency formatting (`$`, `,`) in `median_house_value`, and verified numeric data types.
2. **Exploratory Data Analysis (EDA):** Computed pairwise correlation heatmaps to identify strong positive drivers (such as `median_income`) and collinear feature clusters.
3. **Feature Engineering & VIF Analysis:** Engineered per-household density ratios and interaction terms, diagnosed multicollinearity via Variance Inflation Factor (VIF), and removed redundant raw count columns.
4. **Preprocessing Pipeline:** Partitioned data into 80% training and 20% test sets (`random_state=42`), applied median imputation and `StandardScaler` to continuous features, and one-hot encoded `ocean_proximity`.
5. **Model Training & Tuning:** Trained baseline OLS alongside Ridge and Lasso models tuned over `alpha` in `[0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]` using 5-fold `GridSearchCV`.
6. **Residual Error Analysis:** Plotted Predicted vs. Actual alignment against the ideal $y = x$ reference line and inspected KDE residual distributions for error symmetry and homoscedasticity.

---

## 5. Model Performance Summary

*Update the bracketed placeholders below with the exact metrics generated in Cell 5 of your notebook:*

| Model | Best Alpha | RMSE ($) | MAE ($) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **OLS (Linear Regression)** | N/A | `[INSERT_RMSE]` | `[INSERT_MAE]` | `[INSERT_R2]` |
| **Ridge Regression (L2)** | `[INSERT_ALPHA]` | `[INSERT_RMSE]` | `[INSERT_MAE]` | `[INSERT_R2]` |
| **Lasso Regression (L1)** | `[INSERT_ALPHA]` | `[INSERT_RMSE]` | `[INSERT_MAE]` | `[INSERT_R2]` |

---

## 6. Repository Structure & How to Run

```text
├── housing_price_prediction.ipynb   # Complete Google Colab regression notebook
├── housing dataset.csv              # California Housing dataset (20,640 rows)
└── README.md                        # Project documentation and summary
```

* **Step 1 — Clone the Repository:** Run `git clone https://github.com/[YOUR_GITHUB_USERNAME]/[YOUR_REPO_NAME].git` in your terminal.
* **Step 2 — Open in Google Colab:** Upload `housing_price_prediction.ipynb` to [Google Colab](https://colab.research.google.com/).
* **Step 3 — Upload Dataset & Execute:** Run **Cell 1** to upload `housing dataset.csv`, then run **Cells 2 through 6** sequentially to reproduce all metrics, heatmaps, and residual distribution curves.