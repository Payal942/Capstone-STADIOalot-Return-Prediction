# Notebooks

This folder contains the Jupyter notebooks used to prepare the STADIOalot e-commerce returns dataset and develop the machine-learning models used for return prediction.

The notebooks follow a sequential machine-learning workflow:

**Raw Dataset → Data Preprocessing → Feature Engineering → Model 1: Random Forest → Model 2: XGBoost → Model Evaluation**

---

## Notebook 1 — Data Preprocessing

The Data Preprocessing notebook prepares the raw synthetic e-commerce dataset for machine-learning analysis.

### Main processing steps

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

### Target variable

The target variable is:

`Return_Status`

This represents whether an order was returned:

* `1` = Returned
* `0` = Not Returned

### Leakage prevention

The notebook removes variables that would not be available when making a prediction before the return occurs, including:

* `Return_Date`
* `Return_Reason`
* `Days_to_Return`
* `Refund_Date`
* `Refund_Amount`

### Input

`data/raw/ecommerce_returns_synthetic_data.csv`

### Output

`data/processed/returns_preprocessed.csv`

---

## Notebook 2 — Feature Engineering

The Feature Engineering notebook uses the cleaned dataset produced by Notebook 1 and transforms the available information into features suitable for machine-learning models.

### Main feature-engineering steps

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

### Temporal features

The notebook creates the following features from `Order_Date`:

* `order_year`
* `order_month`
* `order_day`
* `order_day_of_week`
* `order_week_of_year`
* `order_is_weekend`

The original `Order_Date` column is removed after these features have been created.

### Commercial features

Where the required source variables are available, the notebook creates:

* `order_gross_value` — estimated as Product Price × Order Quantity
* `discount_fraction` — discount represented as a fraction
* `discount_amount_est` — estimated monetary discount

### Customer features

Customer age is transformed into categorical age bands:

* `<18`
* `18-24`
* `25-34`
* `35-44`
* `45-54`
* `55-64`
* `65+`

### Identifier removal

Identifier variables are removed because they identify individual records or entities rather than representing generalisable predictive characteristics.

Examples include:

* `Order_ID`
* `Product_ID`
* `User_ID`
* `Customer_ID`
* `Transaction_ID`

### Input

`data/processed/returns_preprocessed.csv`

### Output

`data/processed/returns_features.csv`

---

## Notebook 3 — Model 1: Random Forest

The Model 1 notebook develops a Random Forest classifier to predict whether an e-commerce order will be returned.

The model uses the feature-engineered dataset produced by Notebook 2.

### Main modelling steps

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

### Preprocessing

Numerical features use:

* Median imputation
* `StandardScaler`

Categorical features use:

* Most-frequent imputation
* `OneHotEncoder`
* `handle_unknown="ignore"`

The preprocessing is fitted only on the training data as part of the modelling pipeline.

### Random Forest configuration

| Parameter           |                Value |
| ------------------- | -------------------: |
| `n_estimators`      |                  300 |
| `max_depth`         |                   12 |
| `min_samples_split` |                    5 |
| `min_samples_leaf`  |                    2 |
| `class_weight`      | `balanced_subsample` |
| `random_state`      |                   42 |
| `n_jobs`            |                   -1 |

### Evaluation

The Random Forest model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Classification report
* Confusion matrix

### Input

`data/processed/returns_features.csv`

### Outputs

**Trained model:**

`models/model1_random_forest.joblib`

**Evaluation metrics:**

`experimental_results/model1_random_forest_metrics.csv`

**Confusion matrix:**

`visualisations/model1_random_forest_confusion_matrix.png`

---

## Notebook 4 — Model 2: XGBoost

The Model 2 notebook develops an XGBoost classifier to predict whether an e-commerce order will be returned.

The XGBoost model uses the same feature-engineered dataset, preprocessing approach and 80/20 stratified train/test split used for Model 1. This provides a consistent basis for comparing the two machine-learning approaches.

### Main modelling steps

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

### Preprocessing

Numerical features use:

* Median imputation
* `StandardScaler`

Categorical features use:

* Most-frequent imputation
* `OneHotEncoder`
* `handle_unknown="ignore"`

The preprocessing is fitted only on the training data as part of the modelling pipeline.

### XGBoost configuration

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

### Evaluation

The XGBoost model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Classification report
* Confusion matrix

### Input

`data/processed/returns_features.csv`

### Outputs

**Trained model:**

`models/model2_xgboost.joblib`

**Evaluation metrics:**

`experimental_results/model2_xgboost_metrics.csv`

**Confusion matrix:**

`visualisations/model2_xgboost_confusion_matrix.png`

---

## Model Development Workflow

The modelling notebooks use the same prepared dataset and evaluation methodology so that the two models can be compared consistently.

```text
Feature-Engineered Dataset
        │
        ▼
returns_features.csv
        │
        ▼
Separate Features (X)
and Target (y)
        │
        ▼
Stratified 80/20 Split
        │
        ├───────────────────────┐
        ▼                       ▼
 Random Forest              XGBoost
   Model 1                   Model 2
        │                       │
        ▼                       ▼
 Predictions + Probabilities
        │                       │
        └───────────┬───────────┘
                    ▼
             Model Evaluation
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Metrics   Confusion  Classification
                  Matrix       Report
                    │
                    ▼
             Saved Model Results
```

---

## Notebook Workflow

The notebooks are designed to be executed in order:

| Notebook                                | Purpose                                          | Input                                           | Output                                                                                         |
| --------------------------------------- | ------------------------------------------------ | ----------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Notebook 1 — 01_DataPreprocessing.ipynb**     | Clean, validate and prepare the raw dataset      | `data/raw/ecommerce_returns_synthetic_data.csv` | `data/processed/returns_preprocessed.csv`                                                      |
| **Notebook 2 — 02_FeatureEngineering.ipynb**    | Create modelling features and remove identifiers | `data/processed/returns_preprocessed.csv`       | `data/processed/returns_features.csv`                                                          |
| **Notebook 3 — 03_Model1_RandomForest.ipynb** | Train and evaluate the Random Forest classifier  | `data/processed/returns_features.csv`           | `models/model1_random_forest.joblib` + `experimental_results/model1_random_forest_metrics.csv` |
| **Notebook 4 — 04_Model2_XGBoost.ipynb**       | Train and evaluate the XGBoost classifier        | `data/processed/returns_features.csv`           | `models/model2_xgboost.joblib` + `experimental_results/model2_xgboost_metrics.csv`             |

Both models also generate their respective confusion-matrix visualisations in the `visualisations/` folder.

The resulting models and evaluation metrics are used in the subsequent model-comparison and reporting stages of the STADIOalot return-prediction project.