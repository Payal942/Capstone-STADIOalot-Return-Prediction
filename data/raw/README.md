# Raw Data

This folder contains the original, unmodified datasets used for the STADIOalot Return Prediction project. It includes the synthetic e-commerce dataset used during the initial model development and the client data extract.
## Datasets

1. Synthetic E-commerce Dataset (SS2)
* **File:** `ecommerce_returns_synthetic_data.csv`
* **Location:** `data/raw/ecommerce_returns_synthetic_data.csv`
* **Description:** Synthetic e-commerce order data used to develop and evaluate a machine learning model for predicting whether an order will be returned.
* **Target Variable:** `Return_Status`

  * `Returned = 1`
  * `Not Returned = 0`

The dataset is stored in its original form and should **not be modified directly**. All cleaning, preprocessing, feature engineering and transformations are performed in the project notebooks and the resulting datasets are saved in the `data/processed/` folder.

### 2. STADIOalot Client Data Extract (SS3 Part B)

The `ClientData/` subfolder contains eight CSV files supplied for the client-data modelling stage. These tables provide information about orders, returns, products, marketplace sellers, customers, customer behaviour, inventory and delivery logistics.

**Location:** `data/raw/ClientData/`

| File                              | Description                                                                   |
| --------------------------------- | ----------------------------------------------------------------------------- |
| `table1_orders_transactions.csv`  | Order and transaction information                                             |
| `table2_returns.csv`              | Return records used to construct the target variable and evaluate predictions |
| `table3_product_catalogue.csv`    | Product catalogue information                                                 |
| `table4_marketplace_sellers.csv`  | Marketplace seller information                                                |
| `table5_customer_information.csv` | Customer demographic and profile information                                  |
| `table6_customer_behaviour.csv`   | Customer behaviour information                                                |
| `table7_inventory_warehouse.csv`  | Inventory and warehouse information                                           |
| `table8_delivery_logistics.csv`   | Delivery and logistics information                                            |

The prediction unit is an order and the target variable is `Return_Status`. An order is labelled `1` if at least one item from that order is returned within the defined 30-day observation window, and `0` otherwise.

The returns table is used for target construction and evaluation rather than as a source of predictive input features. Information that would only be available after the prediction point must be excluded from model training to prevent data leakage.



## Data Workflow

### SS2 — Synthetic Dataset

```text
data/raw/
└── ecommerce_returns_synthetic_data.csv
            |
            v
Notebook 1 — Data Preprocessing
            |
            v
data/processed/
└── returns_preprocessed.csv
            |
            v
Notebook 2 — Feature Engineering
            |
            v
data/processed/
└── returns_features.csv
```

### SS3 Part B — Client Data

```text
data/raw/
└── ClientData/
    ├── table1_orders_transactions.csv
    ├── table2_returns.csv
    ├── table3_product_catalogue.csv
    ├── table4_marketplace_sellers.csv
    ├── table5_customer_information.csv
    ├── table6_customer_behaviour.csv
    ├── table7_inventory_warehouse.csv
    └── table8_delivery_logistics.csv
            |
            v
SS3_PartB.ipynb — Dataset Loading and Inspection
            |
            v
Data Preprocessing and Order-Level Feature Engineering
            |
            v
Training, Validation and Testing
            |
            v
XGBoost Model Training and Evaluation
            |
            v
Results Visualisation and Performance Reporting
```

## Data Handling

All raw datasets are retained in their original form to support reproducibility and traceability. Raw files should not be modified directly. Data inspection, cleaning, transformations and feature engineering should be performed on in-memory copies or through the project notebooks.

For the SS2 synthetic dataset, the processing workflow includes:

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

Processed SS2 datasets are saved separately under `data/processed/`.

For the SS3 client data, the workflow includes:

* Loading and inspecting the eight client tables
* Checking data quality, missing values, duplicates and data types
* Identifying appropriate keys for joining the tables
* Constructing the order-level target variable
* Aggregating item-level and related records to the order level where required
* Excluding post-prediction information and other potential leakage variables
* Engineering predictive features
* Preparing training, validation and test datasets
* Training and evaluating the selected XGBoost model
* Visualising model performance

Only information available at the intended prediction point should be used as a predictive feature.

## Reproducibility

The `data/raw/` directory serves as the source-data layer of the project. Keeping the original datasets unchanged helps ensure that preprocessing and modelling can be repeated consistently.

Processed datasets, trained models, evaluation metrics and visualisations should be stored separately in their designated project directories.

**Note:** The client data extract should only be committed to the Git repository if its distribution and inclusion are permitted by the project or course requirements. If the files contain restricted or confidential information, keep them out of version control and document how they should be obtained.
