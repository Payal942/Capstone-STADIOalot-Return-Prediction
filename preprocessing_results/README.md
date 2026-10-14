# SS3 Preprocessing Output

## Overview

The `ss3_preprocessing_output.csv` file contains the prepared client-data dataset generated during the preprocessing stage of `SS3_Python.ipynb` for the STADIOalot Return Prediction project. The dataset is derived from the eight supplied client tables and is prepared for subsequent feature processing, XGBoost model training and return-prediction evaluation.

## Source Data

The preprocessing workflow uses the following input files from `data/raw/ClientData/`:

| File | Description |
|---|---|
| `table1_orders_transactions.csv` | Orders and transaction information |
| `table2_returns.csv` | Return records |
| `table3_product_catalogue.csv` | Product catalogue information |
| `table4_marketplace_sellers.csv` | Marketplace seller information |
| `table5_customer_information.csv` | Customer information |
| `table6_customer_behaviour.csv` | Customer behaviour information |
| `table7_inventory_warehouse.csv` | Inventory and warehouse information |
| `table8_delivery_logistics.csv` | Delivery and logistics information |

## Output File

**Filename:** `ss3_preprocessing_output.csv`

**Purpose:** Stores the integrated and preprocessed client dataset produced by the notebook.

The preprocessing workflow may include:

- Loading and inspecting the client source tables.
- Checking data types, missing values, duplicate records and join keys.
- Combining relevant tables using their relationships.
- Aggregating item-level information where necessary to establish an order-level dataset.
- Preparing relevant variables for subsequent feature engineering and modelling.
- Identifying and excluding information that would cause target leakage or would not be available at the intended prediction point.

The exact transformations and resulting columns are determined by the implementation in `SS3_Python.ipynb`.

## Target Variable

The modelling target is `Return_Status`, encoded as:

| Value | Meaning |
|---|---|
| `0` | Not Returned |
| `1` | Returned |

The objective is to predict whether an order will be returned using information available at or before the intended prediction point. Post-order information, such as return dates, return reasons and refund details, should not be used as predictive features when these details are unavailable at prediction time.

## Role in the Modelling Workflow

The preprocessing output provides an intermediate dataset for the subsequent stages of the client-data modelling workflow:

1. Load and integrate the eight client tables.
2. Preprocess the integrated dataset.
3. Save the resulting dataset as `ss3_preprocessing_output.csv`.
4. Prepare the modelling features and target variable.
5. Fit the preprocessing component and train the XGBoost classifier.
6. Evaluate the model using classification metrics and visualisations.
7. Save the trained model and evaluation results.

## Related Artefacts

The notebook also generates or uses the following artefacts, depending on its configured execution steps:

| Artefact | Purpose |
|---|---|
| `models/ss3_preprocessor.joblib` | Stores the fitted preprocessing component for reuse. |
| `models/ss3_xgboost_return_prediction.joblib` | Stores the trained XGBoost return-prediction model. |
| `experimental_results/ss3_xgboost_metrics.csv` | Stores the model evaluation metrics. |
| `experimental_results/ss3_xgboost_confusion_matrix.csv` | Stores the confusion matrix results. |
| `experimental_results/ss3_xgboost_training_history.csv` | Stores the recorded model training history. |

