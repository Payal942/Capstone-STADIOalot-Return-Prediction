# Capstone Project: Predictive Demand Forecasting for STADIOalot
A Data Science capstone project focused on developing predictive demand forecasting models for STADIOalot to identify demand patterns, improve forecasting accuracy and support data-driven inventory and operational decisions.

**Student Name:** Payal.S. Morar  
**Student Number:** 23301266  
**Institution:** School of Information Technology, STADIO Higher Education  
**Due Date:** 26 October 2026  

## Table of Contents

1. [Project Overview](#project-overview)
2. [Motivation for the Study](#motivation-for-the-study)
3. [Problem Statement](#problem-statement)
4. [Data Request](#data-request)
5. [Literature Review](#literature-review)
6. [Modelling](#modelling)
   - [Requirements](#requirements)
   - [Data Preprocessing](#data-preprocessing)
   - [Feature Engineering](#feature-engineering)
   - [Model 1 – Random Forest](#model-1--random-forest)
   - [Model 2 – XGBoost](#model-2--xgboost)
7. [Model Performance Evaluation](#model-performance-evaluation)
   - [Model 1 Performance Evaluation](#model-1-performance-evaluation)
   - [Model 2 Performance Evaluation](#model-2-performance-evaluation)
   - [Model Comparison](#model-comparison)
8. [Client Recommendation](#client-recommendation)
9. [Client-Data Return Prediction Workflow](#client-data-return-prediction-workflow)
10. [Project Presentation](#project-presentation)
11. [Repository Structure](#repository-structure)
12. [RAAIDD Log](#raaidd-log)
   - [Risks](#risks)
   - [Actions](#actions)
   - [Assumptions](#assumptions)
   - [Issues](#issues)
   - [Decisions](#decisions)
   - [Dependencies](#dependencies)

## Project Overview

This Capstone project focuses on STADIOalot and investigates how Data Science can be used to support the company's objective of reducing avoidable product returns. STADIOalot operates at significant scale, with approximately 4.1 million active customers and 58 million orders per year. However, the company faces profitability and operational challenges, with revenue of approximately R38 billion and an operating margin of only 1.9%. The project therefore focuses on using historical customer, order, product, seller and operational data to identify patterns associated with product returns and develop a predictive approach that can identify orders that are more likely to be returned.

---

## Motivation for the Study

STADIOalot is a large South African e-commerce business with approximately 4.1 million active customers and 58 million orders annually. Despite generating approximately R38 billion in revenue, the organisation has an operating margin of only 1.9%, making operational efficiency critical to profitability. A key challenge is the increasing rate of product returns, which has risen from 11% to 15% overall and from 22% to 29% for apparel and shoes.


The increasing return rate creates additional fulfilment, logistics, processing and administrative costs and places pressure on STADIOalot's already thin margins. The issue also affects customers, marketplace sellers and the efficiency of the organisation's logistics operations. Importantly, the briefing pack indicates that return behaviour may be predictable from factors such as products, sellers, size-and-colour combinations and customer purchasing patterns. However these patterns are not currently being used proactively before an order is confirmed. This creates an opportunity to shift from primarily managing returns after they occur towards identifying potentially high-risk orders before the cost of a return is incurred. Addressing this issue is important as continued growth in avoidable returns could further reduce profitability and undermine STADIOalot's 2030 strategy to break the returns spiral and improve the economics of each parcel.


Data Science methods can assist by using STADIOalot's existing transactional, product, customer-behaviour, seller and historical returns data to identify patterns associated with return risk. A supervised machine learning can be used to estimate the probability that an order will be returned using information available at or before the point of purchase. These predictions could potentially support targeted interventions, such as improving product or sizing information, identifying products or sellers with higher return risk and supporting seller-performance reviews.


Therefore, this study seeks to generate evidence that may support predictive and proactive decision-making around customer returns. The findings could help STADIOalot identify high-risk orders and the factors associated with returns, providing evidence to support strategies aimed at reducing potentially avoidable returns, improving operational efficiency and contributing to greater profitability.

---

## Problem Statement

Despite STADIOalot's substantial annual order volume, the organisation is confronted with an escalating product return rate, currently standing at 15% overall and 29% for apparel and shoes, which places further pressure on its already thin 1.9% operating margin. Consequently, it remains analytically unclear whether historical transaction records, product catalogue attributes and customer behavioural signals can be systematically leveraged to predict the probability of a product return at the point of purchase. Although STADIOalot records extensive information about customers, orders, products, sellers and historical returns, potentially avoidable returns are currently identified primarily after the purchase has occurred. The briefing pack indicates that return behaviour is influenced by factors such as product characteristics, sellers, size-and-colour combinations and customer purchasing patterns, suggesting that return risk may be predictable before an order is completed.

Therefore, this study aims to develop and evaluate a predictive machine learning classification model using STADIOalot's order, product catalogue and clickstream data to identify transactions at a high risk of return. The project will investigate which customer, product, seller, behavioural and transactional characteristics are associated with return risk and determine whether these characteristics can be used to produce reliable return-risk predictions. The intended outcome is to generate evidence that can support proactive, data-driven interventions, enabling STADIOalot to identify high-risk orders earlier, investigate appropriate preventative strategies and potentially reduce avoidable returns and their associated fulfilment and logistics costs.

---

## Data Request

The data request consists of eight related tables. Each table provides information required to understand factors associated with product returns and to develop predictive features for the machine learning models.

| Table | Purpose |
|---|---|
| **Orders and Transactions** | Provides information about customer orders, products purchased, sellers, prices, promotions, payment methods and delivery options. This forms the core transactional data for the prediction model. |
| **Returns** | Provides information about returned orders, including return reasons, returned quantities, refund outcomes and return costs. This table is primarily used to define the target variable (`Returned = 1/0`) and evaluate the model. |
| **Product Catalogue** | Provides product characteristics such as category, subcategory, brand, size, colour, price, ratings and review information. These attributes can help identify products associated with higher return risk. |
| **Marketplace Sellers** | Provides seller-related information such as seller ratings, tenure, shipping performance, late shipment rates and complaints. This can help determine whether seller characteristics are associated with product returns. |
| **Customer Information** | Provides customer-level characteristics and historical purchasing information, including customer tenure, average order value, historical spending and previous returns. These variables can help identify customer-level return patterns. |
| **Customer Behaviour** | Provides information about customer shopping activity, such as sessions, product views, add-to-basket activity, previous orders and previous returns. This can help identify behavioural patterns associated with return risk. |
| **Inventory and Warehouse** | Provides information about stock availability, stock age, distribution centres and historical fulfilment performance. These variables can help investigate whether inventory and operational factors are associated with returns. |
| **Delivery and Logistics** | Provides information about delivery options, delivery location, route distance and historical delivery performance. These variables can help assess whether logistics and delivery factors contribute to return risk. |

The complete data request, including **column names, data types, examples and additional requirements**, is provided in the Data Request document located in the `docs/` folder.

**Data Request Document:** `docs/Data_Request.pdf`

[Data_Request.pdf](docs/Data_Request.pdf)

---

## Literature Review

A literature review was conducted to identify existing research relevant to predicting e-commerce product returns. Three publications were reviewed to understand commonly used modelling approaches, preprocessing techniques, datasets and findings.

* **Returnformer (Cao, Zhang & Li, 2026):** Uses graph-based learning and transformer techniques to predict product returns using customer-product relationships.
* **Proactive Return Prediction (Ma & Wang, 2024):** Uses a heterogeneous graph neural network to predict returns in online fashion retail.
* **Kalidindi, Narkedamilly & Saladi (2026):** Evaluates predictive models for e-commerce returns and provides a publicly available synthetic dataset that can be used for modelling.

The literature supports the use of machine-learning classification to identify customers or orders at risk of returning products before the return occurs. The reviewed studies also informed the project's preprocessing, feature engineering, model comparison and evaluation approach.

The complete literature review tables and dataset description are provided in:
`docs/Literature_Review.pdf`

[Literature_Review.pdf](docs/Literature_Review.pdf)

---

## Modelling

Two supervised machine learning classification models are used to predict whether an e-commerce order is likely to be returned. The modelling workflow consists of:

1. Raw dataset loading
2. Data preprocessing
3. Feature engineering
4. Model 1 training and inference
5. Model 2 training and inference
6. Model evaluation and comparison

The target variable used for prediction is `Return_Status`, where:

- `Returned = 1`
- `Not Returned = 0`

Post-order information that would not be available at the prediction point is excluded to reduce the risk of data leakage.

---

## Requirements

The Python packages required to run the preprocessing, feature engineering and modelling notebooks/scripts are provided in the **`experimental_setup/`** folder.

The requirements file is located at:

`experimental_setup/requirements.txt`

The requirements file contains the dependencies required for the project environment, including the libraries used for data processing, preprocessing, machine learning, model persistence and evaluation.

To install the project dependencies, run:

```bash
pip install -r experimental_setup/requirements.txt
```

The `experimental_setup/` folder also contains:

* `requirements.txt` – Python libraries and package versions required for the project.
* `configuration.md` – Experimental configuration, modelling settings and other relevant setup information.
* `README.md` – Documentation describing the project environment and experimental setup.


## Data Preprocessing

The preprocessing stage prepares the raw e-commerce returns dataset for modelling. The preprocessing workflow includes data inspection, data-type handling, missing-value treatment, duplicate checks, categorical-value preparation, removal of inappropriate identifier fields and separation of the target variable from the predictive features.

The preprocessing documentation provides the full explanation of the preprocessing decisions, including the reason for each transformation.

**Preprocessing documentation:**  
`docs/Preprocessing.MD`

For instructions on running the preprocessing workflow and understanding the preprocessing techniques used, see: [Preprocessing.MD](docs/Preprocessing.MD)

**Preprocessing notebook/script:**  
`notebooks/01_preprocessing.ipynb`

**Input dataset:**  
`data/raw/ecommerce_returns_synthetic_data.csv`

**Processed dataset:**  
`data/processed/returns_preprocessed.csv`

## Feature Engineering

Feature engineering transforms the cleaned data into predictive variables suitable for machine learning. The feature engineering workflow includes the creation of transactional, temporal and derived numerical features. Examples include:

- Order year
- Order month
- Order day
- Day of week
- Week number
- Weekend indicator
- Order gross value
- Discount fraction
- Estimated discount amount
- Age band

Identifier-like columns that do not provide meaningful predictive information are excluded from the modelling features.

**Feature Engineering documentation:**  
`docs/FeatureEngineering.MD`

For the complete feature engineering methodology and instructions for running the notebook/script, see: [FeatureEngineering.MD](docs/FeatureEngineering.MD)

**Feature Engineering notebook/script:**  
`notebooks/02_feature_engineering.ipynb`

**Input:**  
`data/processed/returns_preprocessed.csv`

**Output:**  
`data/processed/returns_features.csv`

## Model 1 – Random Forest

Model 1 uses a **Random Forest Classifier** to predict whether an order will be returned. The model uses a stratified 80/20 train-test split and preprocessing pipelines for numerical and categorical features. The Random Forest configuration is:

| Hyperparameter | Value |
|---|---:|
| `n_estimators` | 300 |
| `max_depth` | 12 |
| `min_samples_split` | 5 |
| `min_samples_leaf` | 2 |
| `class_weight` | `balanced_subsample` |
| `random_state` | 42 |
| `n_jobs` | -1 |

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

**Model 1 documentation:**  
`docs/Model1.MD`

For the complete Model 1 methodology, hyperparameters, training process and inference instructions, see: [Model1.MD](docs/Model1.MD)

**Model 1 notebook/script:**  
`notebooks/03_model1_random_forest.ipynb`

**Trained model:**  
`models/model1_random_forest.joblib`

**Model results:**  
`experimental_results/model1_random_forest_metrics.csv`

## Model 2 – XGBoost

Model 2 uses an **XGBoost Classifier** to predict whether an order will be returned.

The model uses the same stratified 80/20 train-test approach to support a consistent comparison with Model 1.

The XGBoost configuration is:

| Hyperparameter | Value |
|---|---:|
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.80 |
| `colsample_bytree` | 0.80 |
| `min_child_weight` | 2 |
| `reg_lambda` | 1.0 |
| `random_state` | 42 |
| `n_jobs` | -1 |

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

**Model 2 documentation:**  
`docs/Model2.MD`

For the complete Model 2 methodology, hyperparameters, training process and inference instructions, see: [Model2.MD](docs/Model2.MD)

**Model 2 notebook/script:**  
`notebooks/04_model2_xgboost.ipynb`

**Trained model:**  
`models/model2_xgboost.joblib`

**Model results:**  
`experimental_results/model2_xgboost_metrics.csv`

---

## Model Performance Evaluation
## Model 1 Performance Evaluation

The Model 1 Performance notebook evaluates the saved Random Forest model developed in the modelling stage.

The performance analysis recreates the same stratified 80/20 test split using `random_state=42` and evaluates the saved model on the test observations. The notebook does not retrain the Random Forest model.

The performance evaluation includes:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Classification report
* Confusion matrix
* Bootstrap confidence intervals

Bootstrap confidence intervals are calculated using 1,000 bootstrap samples and a 95% confidence level.

**Model 1 Performance documentation:**

`docs/Model1Performance.MD`

The complete Model 1 performance methodology and results are documented in: [Model1Performance.MD](docs/Model1Performance.MD)

**Model 1 Performance notebook:**

`notebooks/05_Model1Performance.ipynb`

**Saved Model 1:**

`models/model1_random_forest.joblib`

**Performance metrics:**

`experimental_results/model1_random_forest_metrics.csv`

**Bootstrap confidence intervals:**

`experimental_results/model1_bootstrap_confidence_intervals.csv`

---

## Model 2 Performance Evaluation

The Model 2 Performance notebook evaluates the saved XGBoost model developed in the modelling stage.

The performance analysis recreates the same stratified 80/20 test split using `random_state=42` and evaluates the saved model on the test observations. The notebook does not retrain the XGBoost model.

The performance evaluation includes:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Classification report
* Confusion matrix
* Bootstrap confidence intervals

Bootstrap confidence intervals are calculated using 1,000 bootstrap samples and a 95% confidence level.

**Model 2 Performance documentation:**

`docs/Model2Performance.MD`

The complete Model 2 performance methodology and results are documented in: [Model2Performance.MD](docs/Model2Performance.MD)

**Model 2 Performance notebook:**

`notebooks/06_Model2Performance.ipynb`

**Saved Model 2:**

`models/model2_xgboost.joblib`

**Performance metrics:**

`experimental_results/model2_xgboost_metrics.csv`

**Bootstrap confidence intervals:**

`experimental_results/model2_bootstrap_confidence_intervals.csv`

---

## Model Comparison

The Model Comparison notebook provides the comparative analysis of Model 1 and Model 2.

The comparison evaluates the saved Random Forest and XGBoost pipelines on the same stratified test observations using `random_state=42`. Neither model is retrained during the comparison stage.

The comparison includes:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix comparison
* ROC curve comparison
* Metric differences
* Bootstrap 95% confidence intervals
* Paired bootstrap comparison
* McNemar's test

The paired bootstrap analysis reports the metric difference as **Model 2 − Model 1** using 1,000 bootstrap samples and a 95% confidence interval.

McNemar's test is used to compare the classification outcomes produced by the two models on the same test observations, using an alpha level of `0.05`.

**Model Comparison documentation:**

`docs/Comparison.MD`

For the complete model comparison methodology and statistical analysis, see: [Comparison.MD](docs/Comparison.MD)

**Model Comparison notebook:**

`notebooks/07_Comparison.ipynb`

**Model comparison metrics:**

`experimental_results/model_comparison_metrics.csv`

**Model comparison bootstrap confidence intervals:**

`experimental_results/model_comparison_bootstrap_confidence_intervals.csv`

**Metric differences:**

`experimental_results/model_metric_differences.csv`

**McNemar's test:**

`experimental_results/mcnemar_test.csv`

---

## Client Recommendation

The client recommendation report evaluates the results and provides recommendations for the deployment and further development of the selected return-prediction model.

The report addresses:

- The comparison between Model 1 (Random Forest) and Model 2 (XGBoost)
- Recommended improvements to the selected model
- Adaptation of the model to the requested SS1 data
- Alignment between the modelling results and the related literature

The complete client recommendation report is provided in: `docs/Client_Recommendation_Report.pdf`

[Client_Recommendation_Report.pdf](docs/Client_Recommendation_Report.pdf)

---
---

## Client-Data Return Prediction Workflow

In addition to the original modelling and model-comparison workflow, the project includes a client-data workflow that applies the selected XGBoost classification approach to the eight supplied STADIOalot client datasets. The workflow integrates relevant order, return, product, seller, customer, behavioural, inventory and logistics information to create a modelling dataset for predicting whether an order will be returned. The objective is to evaluate the use of available client information for return-risk prediction while ensuring that predictive features are appropriate for the intended prediction point.

### Client Data Sources

The workflow uses the following eight source files located in `data/raw/ClientData/`:

| File | Description |
|---|---|
| `table1_orders_transactions.csv` | Order and transaction information |
| `table2_returns.csv` | Historical return information used to establish the target variable |
| `table3_product_catalogue.csv` | Product attributes and catalogue information |
| `table4_marketplace_sellers.csv` | Marketplace seller information |
| `table5_customer_information.csv` | Customer characteristics and purchasing information |
| `table6_customer_behaviour.csv` | Customer shopping and behavioural information |
| `table7_inventory_warehouse.csv` | Inventory and warehouse information |
| `table8_delivery_logistics.csv` | Delivery and logistics information |

These datasets are integrated according to their relevant identifiers and relationships. Item-level information is aggregated where necessary to support an order-level prediction task.

### Data Preprocessing

The preprocessing stage prepares the integrated client data for modelling. It includes data inspection, data-type handling, missing-value treatment, relevant dataset integration and preparation of the modelling variables. The workflow also considers the prediction point when selecting features. Information that becomes available only after a return occurs, such as return dates, return reasons and refund details, must not be used as predictive features if it would not be available when the prediction is generated.

**Notebook:**

`notebooks/SS3_Python.ipynb`

**Preprocessing output:**

`data/processed/ss3_preprocessing_output.csv`

The preprocessing output provides an intermediate dataset for subsequent model preparation and evaluation.

### Target Variable

The target variable is `Return_Status`, encoded as follows:

| Value | Meaning |
|---|---|
| `0` | Not Returned |
| `1` | Returned |

The model aims to identify orders that are more likely to be returned using information available at or before the intended prediction point.

### Model Training and Evaluation

An XGBoost classifier is trained using the prepared client dataset. The modelling workflow uses a stratified 80/10/10 training, validation and test split to allocate observations for model fitting, validation and final evaluation.

The evaluation includes the following classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The workflow also records confusion-matrix results and training history to support performance assessment and documentation.

### Saved Model Artefacts

The trained model and fitted preprocessing component are saved separately so that they can be loaded and reused.

| File | Purpose |
|---|---|
| `models/ss3_xgboost_return_prediction.joblib` | Stores the trained XGBoost return-prediction model |
| `models/ss3_preprocessor.joblib` | Stores the fitted preprocessing component used to prepare model inputs |

The fitted preprocessor and trained model must be used with the appropriate feature structure and transformations expected during training.

### Client-Data Experimental Results

The workflow generates the following CSV files in `experimental_results/`:

| File | Purpose |
|---|---|
| `ss3_xgboost_metrics.csv` | Records the XGBoost classification evaluation metrics |
| `ss3_xgboost_confusion_matrix.csv` | Records the confusion-matrix results |
| `ss3_xgboost_training_history.csv` | Records the available model training-history information |

These outputs provide a persistent record of model performance and training behaviour. They complement the existing Random Forest, XGBoost and statistical comparison results documented elsewhere in this repository.

### Visualisations

The modelling workflow generates 12 visualisations to support the assessment of data quality, class distribution, XGBoost training behaviour, feature importance, model performance, and potential target leakage. These visualisations are stored in the `visualisations/` directory.

| Visualisation | File | Description |
|---|---|---|
| Visualisation 1: Training and Validation Log Loss | `visualisation_1_training_validation_logloss.png` | Displays training and validation log loss across boosting iterations to assess learning progress and identify potential overfitting. |
| Visualisation 2: Performance Across Data Splits | `visualisation_2_performance_splits.png` | Compares model evaluation metrics across the training, validation and unseen-test datasets to assess consistency and generalisation. |
| Visualisation 3: Unseen-Test Confusion Matrix | `visualisation_3_confusion_matrix.png` | Shows correct and incorrect classifications of returned and non-returned orders on the unseen-test dataset, including false positives and false negatives. |
| Visualisation 4: Unseen-Test ROC Curve | `visualisation_4_test_roc.png` | Illustrates the model's ability to distinguish between returned and non-returned orders across different classification thresholds. |
| Visualisation 5: ROC Curves Across Data Splits | `visualisation_5_roc_splits.png` | Compares ROC curves across the training, validation and unseen-test datasets to examine discrimination performance across the data splits. |
| Visualisation 6: Return Class Imbalance | `visualisation_6_class_imbalance.png` | Displays the distribution of returned and non-returned orders to identify class imbalance and inform the interpretation of evaluation metrics. |
| Visualisation 7: Missing Values Across Client Tables | `visualisation_7_missing_values.png` | Summarises missing-value patterns across the supplied client tables to support data-quality assessment and preprocessing decisions. |
| Visualisation 8: XGBoost Feature Importance | `visualisation_8_feature_importance.png` | Identifies the features ranked highest by the fitted XGBoost model's selected feature-importance measure. |
| Visualisation 9: Performance by Order Month | `visualisation_9_performance_by_month.png` | Examines model performance across order months to identify potential temporal variation in predictive performance. |
| Visualisation 10: Target Leakage Analysis | `visualisation_10_target_leakage_analysis.png` | Examines return rates across delivery-related categories, including on-time status, delivery attempts and delivery timing, to investigate potential leakage risks. |
| Visualisation 11A: Leakage Variables and Returns | `visualisation_11a_leakage_vs_returns.png` | Further investigates the relationship between delivery-related variables and return status, considering whether the variables are available at the intended prediction point. |
| Visualisation 11B: Model Performance With vs Without Leakage | `visualisation_11b_model_performance_comparison.png` | Compares experimental XGBoost results for a baseline feature set and alternative feature sets containing individual or combined delivery-related variables, using ROC-AUC, PR-AUC, precision, recall and F1-score. |


### Reproducibility

To reproduce the client-data workflow:

1. Ensure that the eight client CSV files are available in `data/raw/ClientData/`.
2. Install the required Python dependencies using `experimental_setup/requirements.txt`.
3. Open and execute `notebooks/SS3_Python.ipynb` in the required execution order.
4. Confirm that the preprocessing output, trained model artefacts and experimental results are saved to their configured paths.
5. Review the generated metrics and visualisations to assess model performance.

Reproducibility depends on the availability of the input datasets, compatible dependencies, consistent preprocessing and the configuration used during model training.

This workflow extends the project by applying the return-prediction approach to the supplied client data while preserving the existing model-development, performance-evaluation and statistical-comparison documentation.

---

## Project Presentation

**Title:** STADIOalot Order Return Prediction | CAP182 Capstone Project

This video presentation demonstrates how machine learning can be used to predict the likelihood of e-commerce orders being returned, using information available at or before order confirmation.

### Presentation Highlights

- Business problem and project objectives
- Client dataset and exploratory data analysis (EDA)
- Data preprocessing and feature engineering
- Class imbalance and data leakage considerations
- XGBoost model development and evaluation
- Business recommendations, limitations and future improvements

**YouTube Video:** [https://youtu.be/oA4Sub04Eeo](https://youtu.be/oA4Sub04Eeo)&#x20;

> **Note:** The model is a baseline for further investigation and requires additional validation before operational deployment.

---

## Repository Structure

The GitHub repository is organised to clearly separate datasets, models, experimental configuration and results, statistical analysis, visualisations, notebooks, scripts and supporting documentation. This structure supports reproducibility and makes it easy to locate the different artefacts produced throughout the project.


```text
STADIOalot-Return-Prediction/
│
├── data/
│   ├── raw/
│   |   ├── ClientData/
│   |   |   ├── table1_orders_transactions.csv
│   |   |   ├── table2_returns.csv
│   |   |   ├── table3_product_catalogue.csv
│   |   |   ├── table4_marketplace_sellers.csv
│   |   |   ├── table5_customer_information.csv
│   |   |   ├── table6_customer_behaviour.csv
│   |   |   ├── table7_inventory_warehouse.csv
│   |   |   └── table8_delivery_logistics.csv
│   │   ├── ecommerce_returns_synthetic_data.csv
│   │   └── README.md
│   └── processed/
│       ├── returns_preprocessed.csv
│       ├── returns_features.csv
│       └── README.md
│
├── docs/
│   ├── Client_Recommendation_Report.pdf
│   ├── Comparison.MD
│   ├── Data_Request.pdf
│   ├── FeatureEngineering.MD
│   ├── Literature_Review.pdf
│   ├── Model1.MD
│   ├── Model1Performance.MD
│   ├── Model2.MD
│   ├── Model2Performance.MD
│   ├── Preprocessing.MD
│   └── README.md
│
├── experimental_results/
│   ├── model1_random_forest_metrics.csv
│   ├── model1_bootstrap_confidence_intervals.csv
│   ├── model2_xgboost_metrics.csv
│   ├── model2_bootstrap_confidence_intervals.csv
│   ├── model_comparison_metrics.csv
│   ├── model_comparison_bootstrap_confidence_intervals.csv
│   ├── model_metric_differences.csv
│   ├── mcnemar_test.csv
│   ├── ss3_xgboost_metrics.csv
│   ├── ss3_xgboost_confusion_matrix.csv
│   ├── ss3_xgboost_training_history.csv
│   └── README.md
│
├── experimental_setup/
│   ├── README.md
│   ├── requirements.txt
│   └── configuration.md
│
├── models/
│   ├── model1_random_forest.joblib
│   ├── model2_xgboost.joblib
│   ├── ss3_preprocessor.joblib
│   ├── ss3_xgboost_return_prediction.joblib
│   └── README.md
│
├── notebooks/
│   ├── 01_Preprocessing.ipynb
│   ├── 02_FeatureEngineering.ipynb
│   ├── 03_Model1_RandomForest.ipynb
│   ├── 04_Model2_XGBoost.ipynb
│   ├── 05_Model1Performance.ipynb
│   ├── 06_Model2Performance.ipynb
│   ├── 07_Comparison.ipynb
│   ├── SS3_PartB.ipynb
│   └── README.md
│
├──preprocessing_results/
│   ├── ss3_preprocessing_output.csv
│   └── README.md
|
├── scripts/
│   └── README.md
│
├── statistical_helpers/
│   └── README.md
│
├── visualisations/
│   ├── model1_random_forest_confusion_matrix.png
│   ├── model2_xgboost_confusion_matrix.png
│   ├── visualisation_1_training_validation_logloss.png
│   ├── visualisation_2_performance_splits.png
│   ├── visualisation_3_confusion_matrix.png
│   ├── visualisation_4_test_roc.png
│   ├── visualisation_5_roc_splits.png
│   ├── visualisation_6_class_imbalance.png
│   ├── visualisation_7_missing_values.png
│   ├── visualisation_8_feature_importance.png
│   ├── visualisation_9_performance_by_month.png
│   ├── visualisation_10_target_leakage_analysis.png
│   ├── visualisation_11a_leakage_vs_returns.png
│   ├── visualisation_11b_model_performance_comparison.png
│   └── README.md
│
├── .gitignore
├── LICENSE
└── README.md
```
    
## Repository Structure

| Folder | What goes there |
|---|---|
| `data/raw/` | Stores original datasets received from the client. Raw data is kept unchanged to preserve the original source data.|
| `data/processed/` | Stores cleaned, transformed and prepared datasets used for analysis and model development.|
| `models/` | Stores trained machine learning models, model artefacts and related files.|
| `experimental_setup/` | Contains the project environment and experimental configuration, including requirements.txt and configuration documentation. |
| `experimental_results/` | Stores model evaluation results, performance metrics, experiment outputs and related findings. |
| `statistical_helpers/` | Contains statistical helper functions, statistical tests and scripts used to support model analysis and comparison. |
| `visualisations/` | Contains visualisation scripts and generated graphs, charts and other visual outputs used for exploratory analysis and model evaluation.|
| `notebooks/` | Contains Jupyter notebooks used for exploratory data analysis, preprocessing, feature engineering, modelling and evaluation. |
| `scripts/` | Contains reusable Python scripts used for data processing, modelling, evaluation and other project tasks. |
| `docs/` | Contains supporting project documentation. |

---

## RAAIDD Log

The RAAIDD log identifies the key **Risks, Actions, Assumptions, Issues, Decisions and Dependencies** associated with the proposed STADIOalot customer-return prediction project. Each entry is specifically linked to the objective of using Data Science to predict the probability that an order will be returned before the order is confirmed.

## Risks

| Risk | Description | Potential Impact | Mitigation |
|---|---|---|---|
| **Data leakage** | Information that becomes available only after an order is placed, such as return outcomes, refund information or post-delivery behaviour, may accidentally be included as predictive features. | The model could appear highly accurate during testing and fail to perform realistically on new orders. | Define the prediction point as the point of order confirmation and conduct a feature-by-feature leakage review before model training. |
| **Inconsistent product data** | STADIOalot has approximately 38 million SKUs and the briefing pack identifies inconsistent product categorisation. | Incorrect or inconsistent categories could reduce model quality and make product-level patterns difficult to interpret. | Profile product data, standardise relevant categories and document all transformations during preprocessing. |
| **Unreliable seller information** | Some seller-supplied fields are identified as unreliable in the briefing pack. | Poor-quality seller information could introduce noise or bias into the model. | Assess seller-data completeness and reliability and exclude or transform variables that do not meet defined quality standards. |
| **Class imbalance** | Returned orders may represent a smaller proportion of total orders than non-returned orders. | A model could achieve high accuracy while performing poorly when identifying actual returns. | Evaluate the target distribution and use precision, recall, F1-score and ROC-AUC in addition to accuracy. |
| **Insufficient predictive performance** | Customer, product and seller characteristics may not provide enough information to reliably predict individual returns. | The model may not provide sufficient value for operational decision-making. | Establish a baseline model and compare multiple suitable classification approaches. |
| **Concept drift** | Customer purchasing behaviour, product trends and seller performance may change over time. | A model trained on older data may become less accurate when applied to newer orders. | Use time-aware validation where appropriate and monitor performance across different periods. |
| **High data volume** | STADIOalot processes approximately 58 million orders per year and maintains extensive behavioural and product data. | Large datasets may increase processing time and computational requirements. | Determine an appropriate modelling sample, optimise data processing and document sampling decisions. |
| **Privacy and data governance** | Customer-level behavioural data may contain personally identifiable information that is not necessary for modelling. | Inappropriate use of personal information could create privacy and governance concerns. | Request only data necessary for the project, minimise personally identifiable information and follow applicable data-governance requirements. |

---

## Actions

| Action | Description | Expected Outcome |
|---|---|---|
| **Define prediction target** | Define the return status of an order as the target variable indicating whether the order is subsequently returned. | A clear and measurable classification target. |
| **Define prediction point** | Establish that predictions must use information available at or before order confirmation. | A realistic modelling scenario and reduced risk of data leakage. |
| **Request required data** | Request order, product, customer-behaviour, seller and historical returns data from the client. | Sufficient data to investigate the return-prediction problem. |
| **Profile the data** | Assess data types, missing values, duplicates, outliers, inconsistent categories and identifier quality. | A clear understanding of data quality and preprocessing requirements. |
| **Integrate datasets** | Join relevant datasets using identifiers such as Order ID, Customer ID, SKU and Seller ID. | An integrated analytical dataset suitable for modelling. |
| **Clean the data** | Address missing, inconsistent, duplicated and invalid records. | A reliable dataset for analysis and modelling. |
| **Engineer features** | Create pre-order variables such as historical customer return rate, product return rate and seller performance indicators. | Predictive variables that represent return-risk patterns. |
| **Conduct exploratory analysis** | Investigate relationships between return behaviour and customer, product, seller and transaction characteristics. | Identification of potentially important predictive patterns. |
| **Develop baseline model** | Establish a simple classification model against which more advanced models can be compared. | A benchmark for evaluating model improvements. |
| **Develop and compare models** | Train multiple suitable supervised classification models. | Evidence of which modelling approach provides the most useful predictions. |
| **Evaluate models** | Assess accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices. | A comprehensive understanding of model performance. |
| **Interpret model results** | Identify variables and patterns associated with higher return risk. | Business insights that complement individual predictions. |
| **Develop recommendations** | Translate findings into potential interventions for STADIOalot. | Evidence-based recommendations for proactive return management. |

---

## Assumptions

| Assumption | Description | Why It Matters |
|---|---|---|
| **Historical returns are accurately recorded** | It is assumed that historical data correctly identifies returned and non-returned orders. | Incorrect target labels could significantly affect model training. |
| **Orders can be linked across datasets** | It is assumed that consistent identifiers such as Order ID, Customer ID, SKU and Seller ID exist across relevant datasets. | Dataset integration depends on reliable identifiers. |
| **Historical data is representative** | It is assumed that historical customer and product behaviour contains patterns relevant to future return behaviour. | The model depends on historical patterns being informative about future outcomes. |
| **Sufficient historical data exists** | It is assumed that enough returned and non-returned orders are available to develop and evaluate a model. | A sufficiently large sample is required for reliable machine learning. |
| **The prediction point can be established** | It is assumed that the business can identify the exact point at which a return-risk prediction would be generated. | This determines which variables can legitimately be used by the model. |
| **Pre-order information is available** | It is assumed that relevant customer, product, seller and transaction information is available before order confirmation. | These variables form the basis of the prediction. |
| **Customer behaviour contains predictive signals** | It is assumed that browsing, purchasing and previous-return behaviour contains information related to future returns. | If meaningful patterns do not exist, predictive performance may be limited. |
| **Seller performance is relevant** | It is assumed that seller characteristics may contribute to return risk. | Seller-level information may help explain differences in return behaviour. |
| **Data can be appropriately standardised** | It is assumed that inconsistent product categories and seller information can be cleaned sufficiently for analysis. | Poor standardisation could reduce the usefulness of product and seller features. |

---

## Issues

At the proposal stage, several potential issues have been identified however cannot yet be confirmed until the requested data is received and assessed.

| Issue | Description | Current Status / Proposed Response |
|---|---|---|
| **Data fragmentation** | Browsing, ordering, warehouse, delivery and returns data may exist across separate systems and may not be fully integrated. | To be investigated during data integration and profiling. |
| **Inconsistent product categorisation** | Product categories may be inconsistent across STADIOalot's large product catalogue. | To be assessed during data cleaning and standardisation. |
| **Unreliable seller-supplied fields** | Some marketplace seller information may not be sufficiently reliable for modelling. | Data-quality checks will determine which fields are suitable. |
| **Unknown feature availability** | It is not yet confirmed which variables are consistently available at the exact point of purchase. | The data request and stakeholder discussions will establish feature availability. |
| **Unknown model performance** | It is not yet known whether the available data will provide sufficient predictive power. | Baseline and comparative modelling will be used to assess feasibility. |

---

## Decisions

| Decision | Rationale |
|---|---|
| **Focus the project on customer-return prediction** | Returns represent a significant business challenge and are a priority within STADIOalot's strategic objectives. |
| **Use return status as the primary target variable** | The business objective is to identify whether an order is likely to result in a return. |
| **Predict at or before order confirmation** | This ensures that the project addresses the problem proactively rather than simply explaining returns after they occur. |
| **Use supervised machine learning** | Historical orders can be labelled according to whether they were subsequently returned, making this suitable for a classification problem. |
| **Exclude post-order information from predictive features** | Post-order variables would create data leakage and would not be available when the prediction is required. |
| **Compare multiple classification models** | Comparing models provides stronger evidence than relying on a single modelling approach. |
| **Use multiple evaluation metrics** | Return prediction requires consideration of the model's ability to identify actual returns, so accuracy alone is insufficient. |
| **Prioritise interpretability alongside predictive performance** | STADIOalot needs to understand factors associated with return risk so that predictions can potentially inform business interventions. |
| **Treat the model as decision support** | Predictions should provide evidence to support business decisions rather than automatically determine how every high-risk order is handled. |

---

## Dependencies

| **Dependencies From** | **Dependencies To** | **Description** | **Relationship** |
|---|---|---|---|
| **Client data access** | **Data profiling** | The required datasets must be supplied before data quality can be assessed. | Data profiling depends on receiving the client data. |
| **Data profiling** | **Data cleaning** | Missing, inconsistent and invalid data must first be identified. | Cleaning actions depend on the results of data profiling. |
| **Data cleaning** | **Data integration** | Key identifiers and data-quality issues need to be addressed before datasets are combined. | Reliable integration depends on appropriately prepared datasets. |
| **Data integration** | **Feature engineering** | Predictive features may require information from multiple integrated datasets. | Feature engineering depends on successful dataset integration. |
| **Feature engineering** | **Exploratory analysis** | Predictive variables need to be constructed before their relationships with return status can be investigated. | Exploratory analysis depends on suitable engineered features. |
| **Exploratory analysis** | **Model development** | Initial data patterns and characteristics inform the modelling approach. | Modelling decisions depend on understanding the data. |
| **Model development** | **Model evaluation** | Models must first be trained before predictive performance can be measured. | Evaluation depends on completed model training. |
| **Model evaluation** | **Model selection** | The final model should be selected using evidence from comparative performance. | Model selection depends on evaluation results. |
| **Model interpretation** | **Business recommendations** | Recommendations should be based on demonstrated predictive patterns and model findings. | Business recommendations depend on interpretable model results. |
| **Stakeholder validation** | **Final recommendations** | Business stakeholders need to confirm whether proposed interventions are operationally realistic. | Final recommendations depend on stakeholder feedback. |

---

## RAAIDD Summary

The RAAIDD analysis demonstrates that the proposed project depends on more than simply developing a machine learning model. Reliable and appropriately structured data, careful prevention of data leakage, suitable feature engineering, comparative model evaluation and interpretation of results are all required to determine whether return prediction can provide meaningful business value.

The RAAIDD log will be reviewed and updated throughout the project lifecycle as new information becomes available, particularly once the requested datasets have been received and assessed.

---