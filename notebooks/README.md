# Notebooks

This folder contains the Jupyter notebooks used to prepare the STADIOalot e-commerce returns dataset, engineer predictive features, develop the machine-learning models, evaluate their performance and compare the two models used for return prediction.

The notebooks follow a sequential machine-learning workflow:

**Raw Dataset → Data Preprocessing → Feature Engineering → Model 1: Random Forest → Model 2: XGBoost → Model Performance Evaluation → Model Comparison**

---

## Notebook Workflow

The notebooks are organised into the following stages:

1. **Data Preprocessing** — Clean and validate the raw dataset.
2. **Feature Engineering** — Create predictive features and remove identifier variables.
3. **Model 1 — Random Forest** — Train the first classification model.
4. **Model 2 — XGBoost** — Train the second classification model.
5. **Model 1 Performance** — Evaluate the saved Random Forest model.
6. **Model 2 Performance** — Evaluate the saved XGBoost model.
7. **Model Comparison** — Compare both models using the same test observations and statistical measures.

---

# Notebook 1 — Data Preprocessing

The Data Preprocessing notebook prepares the raw synthetic e-commerce dataset for machine-learning analysis.

## Main processing steps

* Loads and inspects the raw dataset
* Standardises column names
* Checks dataset structure, missing values and duplicates
* Identifies and encodes the target variable `Return_Status`

  * `Returned` = 1
  * `Not Returned` = 0
* Removes duplicate records
* Identifies and removes target-leakage variables that contain information only available after a return occurs
* Converts relevant date columns, including `Order_Date`, to datetime format
* Performs numeric data-quality checks
* Identifies invalid negative values in relevant numeric variables
* Removes rows with missing target values
* Performs final validation checks
* Exports the cleaned dataset

## Target variable

The target variable is:

`Return_Status`

This represents whether an order was returned:

* `1` = Returned
* `0` = Not Returned

## Leakage prevention

The notebook removes variables that would not be available when making a prediction before the return occurs, including:

* `Return_Date`
* `Return_Reason`
* `Days_to_Return`
* `Refund_Date`
* `Refund_Amount`

## Input

`data/raw/ecommerce_returns_synthetic_data.csv`

## Output

`data/processed/returns_preprocessed.csv`

---

# Notebook 2 — Feature Engineering

The Feature Engineering notebook uses the cleaned dataset produced by Notebook 1 and transforms the available information into features suitable for machine-learning models.

## Main feature-engineering steps

* Loads the preprocessed dataset
* Verifies the `Order_Date` datetime field prepared during preprocessing
* Creates temporal features from `Order_Date`
* Creates estimated commercial-value features
* Creates discount-related features
* Creates customer age bands
* Identifies and removes identifier columns
* Performs target validation
* Generates a feature inventory and missing-value report
* Performs final dataset validation
* Exports the feature-engineered dataset

## Temporal features

The notebook creates the following features from `Order_Date`:

* `order_year`
* `order_month`
* `order_day`
* `order_day_of_week`
* `order_week_of_year`
* `order_is_weekend`

The original `Order_Date` column is removed after these features have been created.

## Commercial features

Where the required source variables are available, the notebook creates:

* `order_gross_value` — estimated as Product Price × Order Quantity
* `discount_fraction` — discount represented as a fraction
* `discount_amount_est` — estimated monetary discount

## Customer features

Customer age is transformed into categorical age bands:

* `<18`
* `18-24`
* `25-34`
* `35-44`
* `45-54`
* `55-64`
* `65+`

## Identifier removal

Identifier variables are removed because they identify individual records or entities rather than representing generalisable predictive characteristics.

Examples include:

* `Order_ID`
* `Product_ID`
* `User_ID`
* `Customer_ID`
* `Transaction_ID`

## Input

`data/processed/returns_preprocessed.csv`

## Output

`data/processed/returns_features.csv`

---

# Notebook 3 — Model 1: Random Forest

The Model 1 notebook develops a Random Forest classifier to predict whether an e-commerce order will be returned.

The model uses the feature-engineered dataset produced by Notebook 2.

## Main modelling steps

* Loads the feature-engineered dataset
* Separates the features (`X`) from the target (`Return_Status`)
* Performs an 80/20 stratified train/test split
* Identifies numerical and categorical features
* Applies numerical preprocessing
* Applies categorical preprocessing
* Builds a complete Scikit-learn modelling pipeline
* Trains the Random Forest classifier
* Generates class predictions and return probabilities
* Calculates evaluation metrics
* Generates a classification report
* Creates and saves a confusion matrix
* Saves the trained model
* Saves the evaluation metrics
* Performs final validation checks

## Preprocessing

Numerical features use:

* Median imputation
* `StandardScaler`

Categorical features use:

* Most-frequent imputation
* `OneHotEncoder`
* `handle_unknown="ignore"`

The preprocessing is fitted only on the training data as part of the modelling pipeline.

## Random Forest configuration

| Parameter           |                Value |
| ------------------- | -------------------: |
| `n_estimators`      |                  300 |
| `max_depth`         |                   12 |
| `min_samples_split` |                    5 |
| `min_samples_leaf`  |                    2 |
| `class_weight`      | `balanced_subsample` |
| `random_state`      |                   42 |
| `n_jobs`            |                   -1 |

## Evaluation

The Random Forest model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Classification report
* Confusion matrix

## Input

`data/processed/returns_features.csv`

## Outputs

**Trained model:**

`models/model1_random_forest.joblib`

**Evaluation metrics:**

`experimental_results/model1_random_forest_metrics.csv`

**Confusion matrix:**

`visualisations/model1_random_forest_confusion_matrix.png`

---

# Notebook 4 — Model 2: XGBoost

The Model 2 notebook develops an XGBoost classifier to predict whether an e-commerce order will be returned.

The XGBoost model uses the same feature-engineered dataset, preprocessing approach and 80/20 stratified train/test split used for Model 1. This provides a consistent basis for comparing the two machine-learning approaches.

## Main modelling steps

* Loads the feature-engineered dataset
* Separates the features (`X`) from the target (`Return_Status`)
* Performs an 80/20 stratified train/test split
* Identifies numerical and categorical features
* Applies numerical preprocessing
* Applies categorical preprocessing
* Builds a complete Scikit-learn modelling pipeline
* Trains the XGBoost classifier
* Generates class predictions and return probabilities
* Calculates evaluation metrics
* Generates a classification report
* Creates and saves a confusion matrix
* Saves the trained model
* Saves the evaluation metrics
* Performs final validation checks

## Preprocessing

Numerical features use:

* Median imputation
* `StandardScaler`

Categorical features use:

* Most-frequent imputation
* `OneHotEncoder`
* `handle_unknown="ignore"`

The preprocessing is fitted only on the training data as part of the modelling pipeline.

## XGBoost configuration

| Parameter          | Value |
| ------------------ | ----: |
| `n_estimators`     |   300 |
| `max_depth`        |     6 |
| `learning_rate`    |  0.05 |
| `subsample`        |  0.80 |
| `colsample_bytree` |  0.80 |
| `min_child_weight` |     2 |
| `reg_lambda`       |   1.0 |
| `random_state`     |    42 |
| `n_jobs`           |    -1 |

The model uses the `binary:logistic` objective for binary classification and `logloss` as the evaluation metric.

## Evaluation

The XGBoost model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Classification report
* Confusion matrix

## Input

`data/processed/returns_features.csv`

## Outputs

**Trained model:**

`models/model2_xgboost.joblib`

**Evaluation metrics:**

`experimental_results/model2_xgboost_metrics.csv`

**Confusion matrix:**

`visualisations/model2_xgboost_confusion_matrix.png`

---

# Notebook 5 — Model 1 Performance Evaluation

The Model 1 Performance notebook evaluates the **saved Random Forest model** developed in Notebook 3.

This notebook is used for the performance analysis and focuses on evaluating the already-trained model rather than retraining it.

## Main steps

* Loads the feature-engineered dataset
* Recreates the stratified 80/20 test split using `random_state=42`
* Loads the saved Random Forest pipeline
* Generates predictions for the test observations
* Generates predicted return probabilities
* Calculates performance metrics
* Produces a classification report
* Calculates bootstrap confidence intervals
* Performs validation checks
* Saves statistical performance results

## Model

`models/model1_random_forest.joblib`

## Input

`data/processed/returns_features.csv`

## Outputs

`experimental_results/model1_random_forest_metrics.csv`

`experimental_results/model1_bootstrap_confidence_intervals.csv`

The notebook also supports the performance documentation contained in:

`docs/Model1Performance.MD`

## Important note

The performance notebook **does not retrain the Random Forest model**. It loads the saved model pipeline and evaluates it on the recreated test set using the same split configuration.

---

# Notebook 6 — Model 2 Performance Evaluation

The Model 2 Performance notebook evaluates the **saved XGBoost model** developed in Notebook 4.

This notebook is used for the performance analysis and focuses on evaluating the already-trained model rather than retraining it.

## Main steps

* Loads the feature-engineered dataset
* Recreates the stratified 80/20 test split using `random_state=42`
* Loads the saved XGBoost pipeline
* Generates predictions for the test observations
* Generates predicted return probabilities
* Calculates performance metrics
* Produces a classification report
* Calculates bootstrap confidence intervals
* Performs validation checks
* Saves statistical performance results

## Model

`models/model2_xgboost.joblib`

## Input

`data/processed/returns_features.csv`

## Outputs

`experimental_results/model2_xgboost_metrics.csv`

`experimental_results/model2_bootstrap_confidence_intervals.csv`

The notebook also supports the performance documentation contained in:

`docs/Model2Performance.MD`

## Important note

The performance notebook **does not retrain the XGBoost model**. It loads the saved model pipeline and evaluates it on the recreated test set using the same split configuration.

---

# Notebook 7 — Model Comparison

The Model Comparison notebook provides the comparative analysis.

It evaluates the saved Random Forest and XGBoost models on the **same stratified test observations** so that their performance can be compared consistently.

## Main steps

* Loads the feature-engineered dataset
* Recreates the common stratified 80/20 train/test split using `random_state=42`
* Loads the saved Random Forest pipeline
* Loads the saved XGBoost pipeline
* Generates predictions from both models
* Generates return probabilities from both models
* Calculates side-by-side performance metrics
* Compares confusion matrices
* Compares ROC curves
* Calculates metric differences
* Calculates bootstrap confidence intervals
* Performs paired bootstrap comparison
* Performs McNemar's test
* Saves the comparison results
* Performs final validation checks

## Models compared

**Model 1 — Random Forest**

`models/model1_random_forest.joblib`

**Model 2 — XGBoost**

`models/model2_xgboost.joblib`

## Performance measures

The comparison includes:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrices
* ROC curves
* Bootstrap 95% confidence intervals
* Paired bootstrap metric differences
* McNemar's test

The paired bootstrap comparison reports the difference as:

**Model 2 − Model 1**

A 95% confidence interval is used to assess the uncertainty around the observed metric difference.

McNemar's test is used to examine whether the two models produce statistically different classification outcomes on the same test observations, using an alpha level of 0.05.

## Important note

The Model Comparison notebook **does not retrain either model**.

It loads:

* `model1_random_forest.joblib`
* `model2_xgboost.joblib`

and evaluates both saved pipelines on the same test observations.

## Outputs

`experimental_results/model_comparison_metrics.csv`

`experimental_results/model_comparison_bootstrap_confidence_intervals.csv`

`experimental_results/model_metric_differences.csv`

`experimental_results/mcnemar_test.csv`

The comparison results are documented in:

`docs/Comparison.MD`

---

# Model Development and Evaluation Workflow

The complete notebook workflow is:

```text
Raw Dataset
     │
     ▼
01_DataPreprocessing.ipynb
     │
     ▼
returns_preprocessed.csv
     │
     ▼
02_FeatureEngineering.ipynb
     │
     ▼
returns_features.csv
     │
     ├───────────────────────────────┐
     ▼                               ▼
03_Model1_RandomForest.ipynb    04_Model2_XGBoost.ipynb
     │                               │
     ▼                               ▼
Saved Random Forest             Saved XGBoost
Model Pipeline                  Model Pipeline
     │                               │
     ▼                               ▼
04_Model1Performance.ipynb     05_Model2Performance.ipynb
     │                               │
     └───────────────┬───────────────┘
                     ▼
              07_Comparison.ipynb
                     │
                     ▼
        Comparative Statistical Analysis
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Metrics    Bootstrap   McNemar's
                   CIs          Test
```

---

# Notebook Summary

| Notebook                         | Purpose                                          | Input                                           | Main Output                                              |
| -------------------------------- | ------------------------------------------------ | ----------------------------------------------- | -------------------------------------------------------- |
| **01_DataPreprocessing.ipynb**   | Clean, validate and prepare the raw dataset      | `data/raw/ecommerce_returns_synthetic_data.csv` | `data/processed/returns_preprocessed.csv`                |
| **02_FeatureEngineering.ipynb**  | Create modelling features and remove identifiers | `data/processed/returns_preprocessed.csv`       | `data/processed/returns_features.csv`                    |
| **03_Model1_RandomForest.ipynb** | Train the Random Forest classifier               | `data/processed/returns_features.csv`           | `models/model1_random_forest.joblib` + Model 1 metrics   |
| **04_Model2_XGBoost.ipynb**      | Train the XGBoost classifier                     | `data/processed/returns_features.csv`           | `models/model2_xgboost.joblib` + Model 2 metrics         |
| **05_Model1Performance.ipynb**   | Evaluate the saved Random Forest model           | `returns_features.csv` + saved Model 1          | Model 1 performance and bootstrap results                |
| **06_Model2Performance.ipynb**      | Evaluate the saved XGBoost model                 | `returns_features.csv` + saved Model 2          | Model 2 performance and bootstrap results                |
| **07_Comparison.ipynb**        | Compare both saved models statistically          | `returns_features.csv` + both saved models      | Comparison metrics, bootstrap results and McNemar's test |

---

# Relationship to Project Documentation

Each notebook is supported by documentation in the project's `docs/` folder.

| Notebook                       | Documentation                |
| ------------------------------ | ---------------------------- |
| `01_DataPreprocessing.ipynb`   | `docs/Preprocessing.MD`      |
| `02_FeatureEngineering.ipynb`  | `docs/FeatureEngineering.MD` |
| `03_Model1_RandomForest.ipynb` | `docs/Model1.MD`             |
| `04_Model2_XGBoost.ipynb`      | `docs/Model2.MD`             |
| `05_Model1Performance.ipynb`   | `docs/Model1Performance.MD`  |
| `06_Model2Performance.ipynb`   | `docs/Model2Performance.MD`  |
| `07_Comparison.ipynb`          | `docs/Comparison.MD`         |

The notebooks provide the executable implementation, while the Markdown documentation explains the methodology, inputs, outputs, evaluation procedures and results.

---

# Reproducibility

The modelling and evaluation workflow uses a fixed `random_state=42` and a stratified 80/20 train-test split.

The performance and comparison notebooks use the saved trained model pipelines rather than retraining the models. This ensures that the reported performance and comparison analyses are based on the models developed during the modelling stage.

The statistical comparison uses:

* 1,000 bootstrap samples
* 95% confidence intervals
* Paired bootstrap differences for Model 2 − Model 1
* McNemar's test with `alpha = 0.05`

The required Python packages are specified in the project's `requirements.txt` file.


---

## Client Data Integration and Return Prediction

This additional workflow applies the selected XGBoost return-prediction approach to the supplied STADIOalot client data. It complements the existing notebooks by integrating the client tables, preparing an order-level dataset, training the model and evaluating its performance.

### 1. Client Data Sources

The workflow uses eight CSV files stored in `data/raw/ClientData/`.

| Table | File path | Description |
|---|---|---|
| Table 1 | `data/raw/ClientData/table1_orders_transactions.csv` | Orders and transaction information |
| Table 2 | `data/raw/ClientData/table2_returns.csv` | Return records |
| Table 3 | `data/raw/ClientData/table3_product_catalogue.csv` | Product catalogue information |
| Table 4 | `data/raw/ClientData/table4_marketplace_sellers.csv` | Marketplace seller information |
| Table 5 | `data/raw/ClientData/table5_customer_information.csv` | Customer information |
| Table 6 | `data/raw/ClientData/table6_customer_behaviour.csv` | Customer behaviour information |
| Table 7 | `data/raw/ClientData/table7_inventory_warehouse.csv` | Inventory and warehouse information |
| Table 8 | `data/raw/ClientData/table8_delivery_logistics.csv` | Delivery and logistics information |

### 2. Data Integration and Preparation

The client-data workflow loads and inspects the eight tables before combining the relevant information into an integrated modelling dataset.

The preparation process includes:

- Inspecting data types, missing values, duplicate records and join keys.
- Combining the tables using the relevant relationships between orders, returns, products, customers, sellers, inventory and delivery information.
- Aggregating item-level information where necessary so that each order represents one prediction instance.
- Preparing `Return_Status` as the binary target variable, where `0` represents an order that was not returned and `1` represents an order that was returned.
- Handling missing values and categorical variables through the modelling preprocessing pipeline.
- Excluding identifiers and information that would not be available at the intended prediction point to reduce the risk of data leakage.

The objective is to predict whether an order will be returned using information available at or before order confirmation.

### 3. Model Training and Evaluation

The integrated client dataset is used to train and evaluate the selected XGBoost return-prediction model.

The workflow uses a stratified 80/10/10 split for training, validation and testing. The test set is reserved for evaluating performance on unseen data.

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification report
- Confusion matrix
- ROC curve and relevant performance visualisations

These metrics provide a broader assessment of the model's ability to identify returned orders and distinguish between returned and non-returned orders.

### 4. Saved Models and Results

The client-data workflow saves its model artefacts and evaluation results in the following locations:

| Artifact | File path |
|---|---|
| Trained XGBoost model | `models/ss3_xgboost_return_prediction.joblib` |
| Fitted preprocessing pipeline | `models/ss3_preprocessor.joblib` |
| Model performance metrics | `experimental_results/ss3_xgboost_metrics.csv` |
| Confusion matrix results | `experimental_results/ss3_xgboost_confusion_matrix.csv` |

These artefacts support reproducibility and allow the trained model and its evaluation results to be reviewed after execution. The filenames should correspond to the paths specified in the actual notebook code.

---

### Execution Requirements

Before running this workflow:

1. Ensure that all eight client CSV files are available in `data/raw/ClientData/`.
2. Install the project dependencies listed in `requirements.txt`.
3. Run the relevant client-data notebooks in the required order.
4. Confirm that the integrated dataset, trained model, metrics and visualisations are generated successfully.
5. Verify that the saved artefacts are located in the documented `models/` and `experimental_results/` directories.

**Note:** The existing notebooks and their documentation remain unchanged. This section documents the additional client-data workflow and its associated outputs.


