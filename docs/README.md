# Documentation

This folder contains supporting documentation for the **STADIOalot Return Prediction** project.

The documents in this folder provide the background, data requirements, research context, and supporting information used during the development of the return prediction solution.

## Contents

| Document                | Description                                                                                                                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Data_Request.pdf`      | Contains the detailed data request developed for the project, including the required data tables, column names, data types, examples, and additional information required from STADIOalot. |
| `Literature_Review.pdf` | Contains the literature review of three relevant publications and the selected public dataset used for the project.                                                                        |
| `Preprocessing.MD`      | Documents the data preprocessing process, including data cleaning, handling of missing values, removal of data leakage, and preparation of the dataset for modelling.                      |
| `FeatureEngineering.MD` | Documents the feature engineering process, including temporal, commercial, customer, and other predictive features created for the return prediction models.                               |
| `Model1.MD`             | Documents Model 1, the Random Forest classifier, including its training process, preprocessing pipeline, hyperparameters, predictions, evaluation metrics, and saved outputs.              |
| `Model2.MD`             | Documents Model 2, the XGBoost classifier, including its training and inference process, preprocessing pipeline, hyperparameters, evaluation metrics, and saved outputs.                   |

## Project Documentation Structure

The supporting documentation follows the main development workflow:

```text
docs/
│
├── Data_Request.pdf
├── Literature_Review.pdf
├── Preprocessing.MD
├── FeatureEngineering.MD
├── Model1.MD
└── Model2.MD
```

### Data and Research Documentation

* **Data Request** — Defines the data requirements for the STADIOalot return prediction problem.
* **Literature Review** — Provides the research background and justification for the selected dataset and modelling approach.

### Technical Documentation

* **Preprocessing** — Explains how the raw dataset was cleaned and prepared.
* **Feature Engineering** — Explains the predictive features created from the processed data.
* **Model 1** — Documents the Random Forest implementation and evaluation.
* **Model 2** — Documents the XGBoost implementation and evaluation.

## Relationship to the Notebooks

The technical documentation corresponds to the project's Jupyter notebooks:

| Documentation           | Associated Notebook            |
| ----------------------- | ------------------------------ |
| `Preprocessing.MD`      | `01_Preprocessing.ipynb`       |
| `FeatureEngineering.MD` | `02_FeatureEngineering.ipynb`  |
| `Model1.MD`             | `03_Model1_RandomForest.ipynb` |
| `Model2.MD`             | `04_Model2_XGBoost.ipynb`      |

The notebooks contain the executable implementation, while the corresponding Markdown documents explain the methodology, implementation decisions, outputs, and results.

## Purpose

The documentation stored in this folder provides supporting information for the development and implementation of the **STADIOalot Return Prediction** solution.

Together, these documents provide:

1. The data requirements and research background.
2. The preprocessing and data preparation methodology.
3. The feature engineering methodology.
4. The implementation and evaluation of Model 1.
5. The implementation and evaluation of Model 2.
6. A clear connection between the technical documentation and executable notebooks.

Additional project documentation, reports, and supporting materials may be added to this folder as the project progresses.
