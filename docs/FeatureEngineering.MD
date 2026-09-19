# Feature Engineering

## 1. Overview

This document describes the feature-engineering stage of the **STADIOalot Return Prediction** project.

The purpose of this stage is to transform the preprocessed dataset into a feature-engineered dataset suitable for subsequent machine-learning modelling.

Feature engineering uses information that would reasonably be available **before the return outcome occurs**. Post-return information is excluded during the earlier preprocessing stage.

The implementation is contained in:

```text
notebooks/02_FeatureEngineering.ipynb
```

The notebook performs the following main tasks:

* Loads the preprocessed dataset
* Inspects the input dataset structure
* Verifies the `Order_Date` datetime column
* Creates temporal features
* Removes the original `Order_Date` column after feature extraction
* Identifies commercial variables
* Creates estimated order gross value
* Creates discount-related features
* Creates customer age bands
* Identifies and removes identifier columns
* Validates the `Return_Status` target
* Generates a feature inventory
* Generates a missing-value report
* Generates final dataset summaries
* Validates the final dataset
* Saves the feature-engineered dataset

---

## 2. Notebook and File Location

The feature-engineering implementation is contained in:

```text
notebooks/02_FeatureEngineering.ipynb
```

The notebook is part of the following processing workflow:

```text
data/raw/
└── ecommerce_returns_synthetic_data.csv
              │
              ▼
01_Preprocessing.ipynb
              │
              ▼
data/processed/
└── returns_preprocessed.csv
              │
              ▼
02_FeatureEngineering.ipynb
              │
              ▼
data/processed/
└── returns_features.csv
```

The preprocessing notebook must be successfully executed before running the feature-engineering notebook.

---

## 3. Input Dataset

The notebook uses the output produced by:

```text
notebooks/01_Preprocessing.ipynb
```

The input file is:

```text
data/processed/returns_preprocessed.csv
```

The notebook checks whether this file exists before attempting to load it.

The input dataset is loaded using:

```python
df = pd.read_csv(
    INPUT_PATH,
    parse_dates=["Order_Date"]
)
```

### Why `parse_dates` is used

`Order_Date` was converted to a datetime during **Notebook 1 — Data Preprocessing**.

When a Pandas DataFrame is exported to CSV, the CSV stores the date values however, does not retain Pandas' internal datetime data type.

Therefore, Notebook 2 uses:

```python
parse_dates=["Order_Date"]
```

when loading the CSV so that `Order_Date` is available as a datetime column for temporal feature extraction.

No new date-cleaning operation is performed in Notebook 2.

---

## 4. Output Dataset

The final feature-engineered dataset is saved to:

```text
data/processed/returns_features.csv
```

The notebook exports the dataset using:

```python
df.to_csv(
    OUTPUT_PATH,
    index=False
)
```

The notebook also generates a save report containing:

* Output filename
* Full output path
* Number of rows saved
* Number of columns saved
* File-existence status

The resulting CSV is used as the input for the subsequent machine-learning modelling stage.

---

## 5. Project Paths

The notebook determines the project root using:

```python
ROOT = Path.cwd()
```

If the current working directory does not contain the `data` directory, the notebook checks the parent directory.

The notebook then constructs the following paths:

```text
Project Root
    │
    ├── data/processed/returns_preprocessed.csv
    │
    └── data/processed/returns_features.csv
```

The output directory is automatically created if it does not already exist.

The notebook displays a path table containing:

| Item           | Description                             |
| -------------- | --------------------------------------- |
| Project Root   | Root directory of the project           |
| Input Dataset  | Full path to `returns_preprocessed.csv` |
| Output Dataset | Full path to `returns_features.csv`     |
| Status         | Whether each path is available or ready |

---

# 6. Input Dataset Inspection

After loading the preprocessed dataset, the notebook generates an input structure report.

The report contains:

* Column name
* Data type
* Number of non-null values
* Number of missing values
* Number of unique values

The notebook also displays the first five records of the input dataset.

This provides a final inspection of the preprocessed data before feature creation begins.

---

# 7. Date Verification

The notebook expects the following date column:

```text
Order_Date
```

The expected date columns are defined as:

```python
expected_date_columns = [
    "Order_Date"
]
```

The notebook checks whether:

1. `Order_Date` exists.
2. The column has a datetime data type.

The verification uses:

```python
pd.api.types.is_datetime64_any_dtype(
    df[column]
)
```

A date verification report is produced containing:

* Column
* Data type
* Datetime verification
* Status

A successfully verified column is reported as:

```text
Ready for feature extraction
```

If the column is missing or does not have the expected datetime type, the notebook reports:

```text
CHECK
```

---

# 8. Temporal Feature Engineering

Temporal features are created from the verified `Order_Date` column.

The notebook extracts six features.

## 8.1 Order Year

Feature:

```text
order_year
```

The feature represents the calendar year of the order.

The value is extracted using:

```python
df[column].dt.year
```

---

## 8.2 Order Month

Feature:

```text
order_month
```

The feature represents the calendar month of the order.

Values range from:

```text
1–12
```

The value is extracted using:

```python
df[column].dt.month
```

---

## 8.3 Order Day

Feature:

```text
order_day
```

The feature represents the day of the month.

Values range from:

```text
1–31
```

The value is extracted using:

```python
df[column].dt.day
```

---

## 8.4 Order Day of Week

Feature:

```text
order_day_of_week
```

The feature is created using:

```python
df[column].dt.dayofweek
```

The encoding is:

```text
0 = Monday
1 = Tuesday
2 = Wednesday
3 = Thursday
4 = Friday
5 = Saturday
6 = Sunday
```

---

## 8.5 Order Week of Year

Feature:

```text
order_week_of_year
```

The notebook uses the ISO calendar week:

```python
df[column].dt.isocalendar().week.astype(int)
```

This represents the ISO week of the year.

---

## 8.6 Order Weekend Indicator

Feature:

```text
order_is_weekend
```

The feature is binary:

```text
0 = Weekday
1 = Weekend
```

The notebook identifies Saturday and Sunday using:

```python
df[column].dt.dayofweek.ge(5)
```

and converts the result to integers.

---

# 9. Temporal Feature Report

After creating the temporal features, the notebook generates a temporal-feature report.

The report records:

* Original date column
* Feature created
* Description

For the current dataset, the expected features are:

```text
order_year
order_month
order_day
order_day_of_week
order_week_of_year
order_is_weekend
```

---

# 10. Removal of Original Date Column

After the temporal features have been created, the original verified date column is removed:

```text
Order_Date
```

The notebook uses:

```python
df = df.drop(
    columns=date_columns,
    errors="ignore"
)
```

The removal report records:

* Columns before removal
* Number of original date columns removed
* Columns after removal

The derived temporal features are retained.

This means the final modelling dataset contains the useful temporal information without retaining the original datetime column.

---

# 11. Commercial Variable Detection

The notebook creates a case-insensitive column lookup helper.

This allows the notebook to search for alternative names for commercial variables.

The lookup helper checks the following variables.

## 11.1 Price

Possible column names:

```text
Product_Price
Price
Unit_Price
```

---

## 11.2 Quantity

Possible column names:

```text
Order_Quantity
Quantity
Qty
```

---

## 11.3 Discount

Possible column names:

```text
Discount_Applied
Discount
Discount_Rate
```

---

## 11.4 Customer Age

Possible column names:

```text
User_Age
Customer_Age
Age
```

The notebook produces a commercial-variable report showing:

* Feature type
* Detected column
* Status

Possible statuses are:

```text
FOUND
NOT FOUND
```

The feature-engineering process is designed to continue when an optional variable is unavailable.

---

# 12. Estimated Order Gross Value

If both a price and quantity column are detected, the notebook creates:

```text
order_gross_value
```

The calculation is:

```text
order_gross_value =
Price × Quantity
```

The relevant variables are first converted to numeric values using:

```python
pd.to_numeric(
    df[column],
    errors="coerce"
)
```

The feature therefore represents the estimated gross value of the order before discount.

The notebook produces an order-value report showing:

* Feature
* Formula
* Status

If the required price or quantity column is unavailable, the feature is not created.

---

# 13. Discount Feature Engineering

If a discount column is detected, the notebook converts it to numeric values.

The notebook then determines whether the discount is represented as a percentage or as a fraction.

## 13.1 Percentage Discount

If the median valid discount value is greater than `1`, the notebook assumes the values represent percentages.

For example:

```text
10 → 0.10
25 → 0.25
50 → 0.50
```

The conversion is:

```python
discount_fraction = discount / 100
```

---

## 13.2 Fractional Discount

If the median valid discount value is not greater than `1`, the notebook treats the discount as an existing fraction.

For example:

```text
0.10
0.25
0.50
```

are retained as fractions.

---

## 13.3 Discount Fraction

The resulting feature is:

```text
discount_fraction
```

The values are constrained to the logical range:

```text
0 ≤ discount_fraction ≤ 1
```

using:

```python
.clip(
    lower=0,
    upper=1
)
```

---

# 14. Estimated Discount Amount

If both a discount column and price column are available, the notebook creates:

```text
discount_amount_est
```

The calculation is:

```text
discount_amount_est =
Price × discount_fraction
```

This represents the estimated monetary discount associated with the product price.

The notebook records the created discount features in a discount report.

---

# 15. Customer Age Bands

If a suitable customer-age column is detected, the notebook converts the age values to numeric values and creates:

```text
age_band
```

The age bands are:

| Age Range | Age Band |
| --------- | -------- |
| 0–17      | `<18`    |
| 18–24     | `18-24`  |
| 25–34     | `25-34`  |
| 35–44     | `35-44`  |
| 45–54     | `45-54`  |
| 55–64     | `55-64`  |
| 65–120    | `65+`    |

The notebook uses:

```python
pd.cut(
    age,
    bins=[
        0,
        17,
        24,
        34,
        44,
        54,
        64,
        120
    ],
    labels=[
        "<18",
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65+"
    ],
    include_lowest=True
)
```

The age-band feature allows the modelling stage to represent customer age as grouped categories.

---

# 16. Age-Band Distribution

When `age_band` is successfully created, the notebook generates an age-distribution report.

The report contains:

* Age band
* Frequency
* Percentage

The percentage is calculated relative to the total number of observations.

This provides a summary of the customer-age groups represented in the dataset.

---

# 17. Identifier Detection

The notebook identifies columns that represent individual entities rather than generalisable predictive characteristics.

The following identifiers are explicitly recognised:

```text
Order_ID
Product_ID
User_ID
Customer_ID
Transaction_ID
```

The notebook also removes any column whose name ends with:

```text
_ID
```

The detection is performed case-insensitively.

For example, a column such as:

```text
Some_ID
```

would also be treated as an identifier.

The notebook generates an identifier report containing:

* Identifier column
* Reason for removal

---

# 18. Identifier Removal

Detected identifier columns are removed using:

```python
df = df.drop(
    columns=identifier_columns,
    errors="ignore"
)
```

The notebook records:

* Columns before removal
* Number of identifier columns removed
* Columns after removal

Identifiers are excluded as they generally represent individual or high-cardinality entities rather than generalisable behavioural characteristics.

Examples include:

```text
Order_ID
Product_ID
User_ID
Customer_ID
Transaction_ID
```

Removing these columns helps prevent the models from relying on observation-specific identifiers rather than meaningful predictive patterns.

---

# 19. Target Validation

The target variable remains:

```text
Return_Status
```

The target is not used to create the engineered predictor variables.

After feature engineering and identifier removal, the notebook validates the target separately.

The target is converted to numeric values using:

```python
df["Return_Status"] = pd.to_numeric(
    df["Return_Status"],
    errors="coerce"
)
```

Rows with missing target values are removed.

The target is then converted to integer format:

```text
0
1
```

The notebook verifies that the target contains only:

```text
0 = Not Returned
1 = Returned
```

This ensures that the final dataset remains suitable for binary classification.

---

# 20. Feature Inventory

The notebook generates a feature inventory for the final dataset.

For every column, the inventory records:

* Feature number
* Feature name
* Data type
* Missing values
* Number of unique values

This provides an overview of the variables available for subsequent modelling.

The inventory includes both retained original variables and newly engineered features.

---

# 21. Missing-Value Analysis

After feature engineering, the notebook generates a missing-value report.

For each column, it calculates:

```text
Column
Missing Values
Missing %
```

The report is sorted by the number of missing values in descending order.

### Important

This notebook **does not perform general missing-value imputation**.

The notebook reports remaining missing values but does not automatically replace them.

Missing predictor values are handled later through the preprocessing pipelines associated with the machine-learning models.

---

# 22. Final Dataset Preview

The notebook displays the first ten records of the final feature-engineered dataset.

This provides a final visual inspection of the dataset before it is exported.

---

# 23. Final Dataset Summary

The notebook generates a final summary containing:

* Final rows
* Final columns
* Duplicate rows
* Missing target values
* Number of temporal features created
* Number of identifier columns removed
* Input dataset filename
* Output dataset filename

This provides an overview of the feature-engineering outcome.

---

# 24. Output Dataset

The final feature-engineered dataset is:

```text
data/processed/returns_features.csv
```

The dataset contains:

### Retained modelling variables

Original variables that remain after preprocessing and identifier removal.

### Temporal features

```text
order_year
order_month
order_day
order_day_of_week
order_week_of_year
order_is_weekend
```

### Commercial features

Where the required source variables are available:

```text
order_gross_value
discount_fraction
discount_amount_est
```

### Customer age feature

Where a customer-age variable is available:

```text
age_band
```

### Target

```text
Return_Status
```

The final dataset does **not** contain:

* Original verified date columns such as `Order_Date`
* Detected identifier columns

---

# 25. Final Validation

Before completion, the notebook performs the following validation checks:

| Validation Check             | Purpose                                         |
| ---------------------------- | ----------------------------------------------- |
| Dataset contains rows        | Confirms data remains after feature engineering |
| Target column exists         | Confirms `Return_Status` is present             |
| Target contains only 0 and 1 | Confirms valid binary target                    |
| No duplicate rows            | Confirms duplicate-free final dataset           |
| No missing target values     | Confirms target completeness                    |
| Output file exists           | Confirms successful export                      |

Each check receives one of the following statuses:

```text
PASS
CHECK
```

The final validation therefore provides a quality-control checkpoint before the dataset is passed to the modelling stage.

---

# 26. Feature-Engineering Workflow

The complete workflow is:

```text
data/processed/returns_preprocessed.csv
                │
                ▼
        Load Preprocessed Data
                │
                ▼
        Inspect Input Structure
                │
                ▼
       Verify Order_Date Datetime
                │
                ▼
       Create Temporal Features
                │
                ├── order_year
                ├── order_month
                ├── order_day
                ├── order_day_of_week
                ├── order_week_of_year
                └── order_is_weekend
                │
                ▼
       Remove Original Order_Date
                │
                ▼
       Identify Commercial Variables
                │
                ▼
       Create order_gross_value
                │
                ▼
       Create Discount Features
                │
                ├── discount_fraction
                └── discount_amount_est
                │
                ▼
          Create age_band
                │
                ▼
       Identify Identifier Columns
                │
                ▼
          Remove Identifiers
                │
                ▼
        Validate Return_Status
                │
                ▼
          Feature Inventory
                │
                ▼
       Missing-Value Analysis
                │
                ▼
        Final Dataset Summary
                │
                ▼
          Final Validation
                │
                ▼
       Save Feature-Engineered Data
                │
                ▼
data/processed/returns_features.csv
```

---

# 27. How to Use the Notebook

## Step 1 — Install Dependencies

From the project root, install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Step 2 — Run Data Preprocessing

Before running feature engineering, execute:

```text
notebooks/01_Preprocessing.ipynb
```

Run the notebook from beginning to end.

This creates:

```text
data/processed/returns_preprocessed.csv
```

---

## Step 3 — Open Feature Engineering

Open:

```text
notebooks/02_FeatureEngineering.ipynb
```

---

## Step 4 — Verify the Input File

The notebook expects:

```text
data/processed/returns_preprocessed.csv
```

If the file cannot be found, the notebook raises a `FileNotFoundError`.

---

## Step 5 — Run the Notebook

Run all cells from beginning to end.

The notebook will:

1. Identify the project root.
2. Locate the preprocessed dataset.
3. Load `returns_preprocessed.csv`.
4. Reload `Order_Date` as a datetime.
5. Verify the date column.
6. Create temporal features.
7. Remove the original date column.
8. Detect commercial variables.
9. Create order-value features.
10. Create discount features.
11. Create customer age bands.
12. Detect and remove identifiers.
13. Validate `Return_Status`.
14. Generate the feature inventory.
15. Generate the missing-value report.
16. Validate the final dataset.
17. Save `returns_features.csv`.

---

# 28. Expected Output

After successful execution, the notebook creates:

```text
data/processed/returns_features.csv
```

The notebook also displays a completion summary containing:

```text
Process
Status
Final Dataset Shape
Target Variable
Output File
```

The expected status is:

```text
COMPLETE
```

---

# 29. Relationship to the Modelling Stage

The feature-engineered dataset produced by this notebook is used by the subsequent machine-learning modelling stage.

The workflow is:

```text
Raw Dataset
    │
    ▼
01_Preprocessing.ipynb
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
    ▼
Machine Learning Modelling
    │
    ├── Random Forest
    └── XGBoost
```

The feature-engineering notebook therefore provides the final structured dataset used as the basis for model development.

---

# 30. Summary

The **Feature Engineering** notebook transforms the preprocessed STADIOalot return dataset into a feature-engineered dataset suitable for machine-learning modelling.

The notebook creates temporal features from `Order_Date`, calculates estimated order gross value, standardises discount information, estimates discount amounts, creates customer age bands, removes identifier columns, validates the target and performs final dataset-quality checks.

The notebook does not perform general missing-value imputation. Remaining missing predictor values are reported and are handled later by the modelling preprocessing pipelines.

The implementation is:

```text
notebooks/02_FeatureEngineering.ipynb
```

The input is:

```text
data/processed/returns_preprocessed.csv
```

The final output is:

```text
data/processed/returns_features.csv
```

This output forms the input dataset for the subsequent predictive modelling stage.
