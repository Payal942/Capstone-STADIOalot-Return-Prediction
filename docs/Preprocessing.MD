# Data Preprocessing

## 1. Overview

This document describes the data preprocessing stage of the **STADIOalot Return Prediction** project.

The purpose of this stage is to inspect and prepare the raw e-commerce return dataset before it is passed to the feature-engineering and machine-learning stages. The preprocessing implementation is contained in:

```text
notebooks/01_Preprocessing.ipynb
```

The notebook performs:

* Dataset loading
* Dataset inspection
* Column-name standardisation
* Target-variable identification
* Target-label encoding
* Target-distribution analysis
* Duplicate detection and removal
* Missing-value inspection
* Target-leakage identification and removal
* Date-column conversion
* Numerical data-quality checks
* Invalid negative-value handling
* Missing-target handling
* Final validation
* Export of the preprocessed dataset

---

## 2. Input Dataset

The notebook reads the raw dataset from:

```text
data/raw/ecommerce_returns_synthetic_data.csv
```

The dataset used for this project is the **Synthetic Dataset for E-Commerce Return Analysis**.

The dataset source is:

```text
https://www.kaggle.com/datasets/sayalikhot21/synthetic-dataset-for-e-commerce-return-analysis
```

The notebook first checks whether the expected input file exists before loading it.

The dataset is loaded using:

```python
df = pd.read_csv(RAW_PATH)
```

The raw dataset is preserved in its original form. No changes are made directly to the file in `data/raw/`.

---

## 3. Output Dataset

After preprocessing, the notebook saves the resulting dataset to:

```text
data/processed/returns_preprocessed.csv
```

The output is created using:

```python
df.to_csv(
    OUTPUT_PATH,
    index=False
)
```

The processed dataset is subsequently used as the input for:

```text
notebooks/02_FeatureEngineering.ipynb
```

---

## 4. Project Paths

The notebook determines the project root using:

```python
ROOT = Path.cwd()
```

If the current working directory does not contain the `data` directory, the notebook checks the parent directory.

The following absolute paths are then constructed:

```text
Project Root
    │
    ├── data/raw/ecommerce_returns_synthetic_data.csv
    │
    └── data/processed/returns_preprocessed.csv
```

The processed-data directory is automatically created if it does not already exist.

The notebook also displays a path table showing:

* Project root
* Raw dataset path
* Processed dataset path
* Current path status

---

## 5. Dataset Inspection

After loading the raw dataset, the notebook generates an initial dataset summary containing:

* Number of rows
* Number of columns
* Memory usage

The notebook also displays the first and last five records.

A dataset structure table is generated containing:

* Column name
* Data type
* Non-null values
* Missing values
* Unique values

This provides an initial understanding of the dataset before preprocessing.

---

## 6. Column-Name Standardisation

The notebook standardises column names by:

* Converting column names to strings
* Removing leading and trailing whitespace
* Replacing spaces with underscores
* Replacing hyphens with underscores

The implementation is:

```python
df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
)
```

A comparison table is displayed showing the original and standardised column names.

This creates consistent column naming for the subsequent preprocessing and feature-engineering stages.

---

## 7. Target Variable Identification

The classification target is:

```text
Return_Status
```

The notebook searches for the target using the following possible column names:

```python
target_candidates = [
    "Return_Status",
    "return_status",
    "ReturnStatus"
]
```

If the target variable cannot be found, the notebook raises an error and displays the available columns.

The target is explicitly identified before target preparation is performed.

---

## 8. Target Variable Encoding

The target variable represents whether an order was returned.

The project uses the following binary classification encoding:

```text
Returned     = 1
Not Returned = 0
```

Before encoding, target values are:

* Converted to strings
* Stripped of leading and trailing whitespace
* Converted to lowercase

The notebook uses the following mapping:

```python
target_mapping = {
    "returned": 1,
    "return": 1,
    "yes": 1,
    "1": 1,

    "not returned": 0,
    "not_returned": 0,
    "no": 0,
    "0": 0
}
```

Any target values that cannot be mapped are identified. If unmapped values are found, the notebook stops with an error.

The final target variable is stored as:

```text
Return_Status
```

and converted to integer format.

---

## 9. Target Distribution

After encoding, the notebook calculates the target distribution.

The distribution table contains:

* Return-status value
* Label
* Frequency
* Percentage

The labels are:

```text
0 = Not Returned
1 = Returned
```

This provides an overview of the class distribution before further preprocessing and modelling.

---

## 10. Duplicate Detection and Removal

The notebook detects duplicate records using all columns.

The number of duplicate rows is calculated using:

```python
duplicate_count = df.duplicated().sum()
```

Duplicate rows are then removed using:

```python
df = (
    df
    .drop_duplicates()
    .reset_index(drop=True)
)
```

A duplicate-removal report records:

* Rows before removal
* Duplicate rows removed
* Rows after removal
* Remaining duplicate rows

This ensures that identical observations are not unnecessarily retained in the processed dataset.

---

## 11. Missing-Value Inspection

The notebook performs missing-value inspection before and after preprocessing.

For each column, the notebook calculates:

```text
Column
Missing Values
Missing %
```

The missing percentage is calculated as:

```python
missing_report["Missing %"] = (
    missing_report["Missing Values"]
    / len(df)
    * 100
)
```

The report is sorted by the number of missing values so that columns containing missing data can be identified.

### Missing-value handling

This notebook **does not perform general missing-value imputation**.

Missing values that remain after preprocessing are retained in the processed dataset and are handled later by the machine-learning preprocessing pipelines.

Therefore, operations such as:

* Median imputation
* Most-frequent categorical imputation

are **not performed by this notebook**.

---

## 12. Target Leakage Identification

The project aims to predict whether an order will be returned using information available before the return occurs.

The notebook therefore identifies variables that may contain information only available after the return process.

The possible leakage variables checked by the notebook are:

```text
Return_Date
Return_Reason
Days_to_Return
Refund_Date
Refund_Amount
```

The notebook checks the corresponding column names specified in the preprocessing code.

The leakage columns actually present in the dataset are stored in:

```python
leakage_columns
```

A leakage report is generated showing:

* Leakage column
* Reason for removal

If no leakage variables are detected, the notebook reports that no post-return variables were present.

---

## 13. Removal of Target Leakage

Any identified leakage columns are removed using:

```python
df = df.drop(
    columns=leakage_columns,
    errors="ignore"
)
```

The notebook records:

* Number of columns before leakage removal
* Number of leakage columns removed
* Number of columns after leakage removal

The purpose of this step is to prevent information that would only become available after the return process from being used as a predictive feature.

Examples include:

```text
Return_Date
Return_Reason
Days_to_Return
Refund_Date
Refund_Amount
```

These variables are not appropriate for a prediction made before the return occurs.

---

## 14. Date-Column Conversion

After leakage removal, the notebook searches for remaining columns whose names contain:

```text
date
```

The identified columns are converted using:

```python
pd.to_datetime(
    df[column],
    errors="coerce"
)
```

A date column is converted only when at least **70% of its values** can successfully be interpreted as dates.

The notebook records:

* Column name
* Percentage of valid dates
* Conversion status

The converted date columns are **retained in the preprocessed dataset**.

They are subsequently used by:

```text
notebooks/02_FeatureEngineering.ipynb
```

for the creation of temporal features such as order year, month, day and day of week.

> **Note:** Saving a Pandas datetime column to CSV preserves the date values however, does not preserve Pandas' internal datetime data type. Notebook 2 therefore reloads the relevant date column using `parse_dates=["Order_Date"]` before creating temporal features.

---

## 15. Numerical Data-Quality Checks

The notebook identifies numerical columns using:

```python
df.select_dtypes(
    include=np.number
)
```

The target variable `Return_Status` is excluded from these checks.

The notebook checks relevant numerical variables whose names contain:

```text
price
amount
quantity
discount
```

for negative values.

These checks are intended to identify potentially invalid values in variables representing:

* Prices
* Monetary amounts
* Quantities
* Discounts

A numerical data-quality report is generated containing:

```text
Column
Negative Values
Action
```

---

## 16. Handling Invalid Negative Values

Where negative values are identified in relevant numerical columns, those values are converted to missing values.

The implementation is:

```python
negative_mask = numeric_values < 0

df.loc[
    negative_mask,
    column
] = np.nan
```

The notebook records:

* Column
* Number of invalid negative values
* Number of values converted to missing

The resulting missing values are not imputed in this notebook.

They remain available for the subsequent machine-learning preprocessing pipelines to handle.

---

## 17. Missing Target Values

The notebook checks for missing values in:

```text
Return_Status
```

Rows with a missing target cannot be used for supervised classification and are therefore removed.

The implementation is:

```python
df = (
    df
    .dropna(subset=["Return_Status"])
    .reset_index(drop=True)
)
```

A target-removal report records:

* Rows before removal
* Rows with missing target
* Rows removed
* Rows after removal

The remaining target values are therefore available for binary classification.

---

## 18. Final Missing-Value Report

After preprocessing, the notebook generates a final missing-value report containing:

```text
Column
Missing Values
Missing %
```

This report provides visibility into any missing values remaining in the preprocessed dataset.

The notebook does not automatically impute these remaining values.

---

## 19. Final Preprocessing Summary

The notebook generates a final preprocessing summary containing:

```text
Final Rows
Final Columns
Duplicate Rows
Missing Target Values
Target Variable
Date Columns Converted
Leakage Columns Removed
```

This provides an overview of the resulting dataset after the preprocessing operations.

---

## 20. Final Dataset Preview

The notebook displays the first ten records of the final preprocessed dataset.

It also displays the final target distribution using:

```text
0 = Not Returned
1 = Returned
```

This allows the dataset to be inspected before export.

---

## 21. Export of the Preprocessed Dataset

The final dataset is saved to:

```text
data/processed/returns_preprocessed.csv
```

The notebook creates the output directory if required and exports the dataset using:

```python
df.to_csv(
    OUTPUT_PATH,
    index=False
)
```

A save report confirms:

* Output file
* Rows saved
* Columns saved
* Whether the file exists

The complete absolute output path is also displayed by the notebook.

---

## 22. Final Validation

The notebook performs the following validation checks:

| Validation Check             | Purpose                                        |
| ---------------------------- | ---------------------------------------------- |
| Dataset contains rows        | Confirms that data remains after preprocessing |
| Target column exists         | Confirms `Return_Status` is present            |
| Target contains only 0 and 1 | Confirms valid binary target encoding          |
| No duplicate rows            | Confirms duplicate removal                     |
| No missing target values     | Confirms the target is complete                |
| Processed file exists        | Confirms successful export                     |

Each validation check is assigned:

```text
PASS
```

or:

```text
CHECK
```

The notebook then displays the final dataset shape, target variable and processed dataset path.

---

## 23. Preprocessing Workflow

The complete preprocessing workflow is:

```text
data/raw/ecommerce_returns_synthetic_data.csv
                    │
                    ▼
              Load Dataset
                    │
                    ▼
            Inspect Dataset
                    │
                    ▼
       Standardise Column Names
                    │
                    ▼
          Identify Target
                    │
                    ▼
       Encode Return_Status
                    │
                    ▼
       Analyse Target Distribution
                    │
                    ▼
          Remove Duplicates
                    │
                    ▼
        Inspect Missing Values
                    │
                    ▼
      Identify Leakage Variables
                    │
                    ▼
        Remove Leakage Variables
                    │
                    ▼
       Convert Suitable Dates
                    │
                    ▼
      Check Numerical Variables
                    │
                    ▼
 Convert Invalid Negative Values
          to Missing Values
                    │
                    ▼
       Remove Missing Targets
                    │
                    ▼
          Final Validation
                    │
                    ▼
              Export Dataset
                    │
                    ▼
data/processed/returns_preprocessed.csv
                    │
                    ▼
notebooks/02_FeatureEngineering.ipynb
```

---

## 24. Notebook Associated With This Document

The implementation described in this document is contained in:

```text
notebooks/01_Preprocessing.ipynb
```

This notebook is the implementation of the preprocessing stage for the STADIOalot Return Prediction project.

---

## 25. How to Run the Notebook

### Step 1 — Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

### Step 2 — Verify the Raw Dataset

Ensure the following file exists:

```text
data/raw/ecommerce_returns_synthetic_data.csv
```

### Step 3 — Start Jupyter

Run:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

### Step 4 — Open the Notebook

Open:

```text
notebooks/01_Preprocessing.ipynb
```

### Step 5 — Run the Notebook

Run the cells from beginning to end.

Successful execution will create:

```text
data/processed/returns_preprocessed.csv
```

---

## 26. Relationship to the Next Stage

The preprocessed dataset produced by this notebook is:

```text
data/processed/returns_preprocessed.csv
```

This file becomes the input for:

```text
notebooks/02_FeatureEngineering.ipynb
```

The feature-engineering stage uses the preprocessed data to create predictive features from information available before the return occurs.

The resulting dataset is:

```text
data/processed/returns_features.csv
```

---

## 27. Summary

The **Data Preprocessing** notebook establishes the first data-preparation stage of the STADIOalot Return Prediction project.

It loads and inspects the raw dataset, standardises column names, prepares the binary target variable, removes duplicate records, identifies missing values, removes potential post-return leakage variables, converts suitable date columns, checks numerical data quality, handles invalid negative values, removes records with missing target values and validates the final dataset.

General missing-value imputation and predictive feature engineering are **not performed in this notebook**. These are handled in later stages of the project.

The final output is:

```text
data/processed/returns_preprocessed.csv
```

which is passed to the feature-engineering notebook for the next stage of the machine-learning workflow.
