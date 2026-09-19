# Raw Data

This folder contains the original, unmodified dataset used for the **STADIOalot Return Prediction** project.

## Dataset

* **File:** `ecommerce_returns_synthetic_data.csv`
* **Location:** `data/raw/ecommerce_returns_synthetic_data.csv`
* **Description:** Synthetic e-commerce order data used to develop and evaluate a machine learning model for predicting whether an order will be returned.
* **Target Variable:** `Return_Status`

  * `Returned = 1`
  * `Not Returned = 0`

The dataset is stored in its original form and should **not be modified directly**. All cleaning, preprocessing, feature engineering and transformations are performed in the project notebooks and the resulting datasets are saved in the `data/processed/` folder.

## Data Workflow

```text
data/raw/
└── ecommerce_returns_synthetic_data.csv
            │
            ▼
Notebook 1 — Data Preprocessing
            │
            ▼
data/processed/
└── returns_preprocessed.csv
            │
            ▼
Notebook 2 — Feature Engineering
            │
            ▼
data/processed/
└── returns_features.csv
```

## Data Handling

The raw dataset is retained as the source version for reproducibility and traceability. The following operations are performed on copies of the data rather than modifying the raw file:

* Data inspection and validation
* Column-name standardisation
* Duplicate detection and removal
* Missing-value handling
* Target-variable preparation
* Data-type conversion
* Removal of return-related data leakage
* Data-quality checks
* Temporal feature engineering
* Commercial feature engineering
* Age-band creation
* Identifier removal

All processed versions are saved separately under `data/processed/`.
