# Documentation

This folder contains supporting documentation for the **STADIOalot Return Prediction** project.

The documents in this folder provide the background, data requirements, research context, methodology, model implementation details, performance evaluation and model comparison information used during the development of the return prediction solution.

---

## Contents

| Document                | Description                                                                                                                                                                                                 |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Data_Request.pdf`      | Contains the detailed data request developed for the project, including the required data tables, column names, data types, examples and additional information required from STADIOalot.                  |
| `Literature_Review.pdf` | Contains the literature review of three relevant publications and the selected public dataset used for the project.                                                                                         |
| `Preprocessing.MD`      | Documents the data preprocessing process, including data cleaning, handling of missing values, removal of data leakage and preparation of the dataset for modelling.                                       |
| `FeatureEngineering.MD` | Documents the feature engineering process, including temporal, commercial, customer and other predictive features created for the return prediction models.                                                |
| `Model1.MD`             | Documents Model 1, the Random Forest classifier, including its training process, preprocessing pipeline, hyperparameters, predictions, evaluation metrics and saved outputs.                               |
| `Model2.MD`             | Documents Model 2, the XGBoost classifier, including its training and inference process, preprocessing pipeline, hyperparameters, evaluation metrics and saved outputs.                                    |
| `Model1Performance.MD`  | Documents the performance evaluation of Model 1, including classification metrics, confusion matrix results, bootstrap confidence intervals and the associated performance analysis notebook.       |
| `Model2Performance.MD`  | Documents the performance evaluation of Model 2, including classification metrics, confusion matrix results, bootstrap confidence intervals and the associated performance analysis notebook.       |
| `Comparison.MD`         | Documents the comparative evaluation of Model 1 and Model 2, including performance metric differences, paired bootstrap confidence intervals, McNemar's test, confusion matrices and ROC curve comparison. |
| `Client_Recommendation_Report.pdf`         | Contains the client recommendation report, including the selected model, recommended model improvements, adaptation requirements for the requested STADIOalot data and alignment with the related literature. |

---

## Project Documentation Structure

The supporting documentation follows the main development and evaluation workflow:

```text
docs/

│
├── Data_Request.pdf
├── Literature_Review.pdf
│
├── Preprocessing.MD
├── FeatureEngineering.MD
│
├── Model1.MD
├── Model2.MD
│
├── Model1Performance.MD
├── Model2Performance.MD
├── Comparison.MD
└── Client_Recommendation.pdf
```

---

## Data and Research Documentation

* **Data Request** — Defines the data requirements for the STADIOalot return prediction problem.

* **Literature Review** — Provides the research background and justification for the selected public dataset and modelling approach.

---

## Technical Documentation

* **Preprocessing** — Explains how the raw dataset was cleaned, validated and prepared for modelling.

* **Feature Engineering** — Explains the predictive features created from the processed data.

* **Model 1** — Documents the Random Forest implementation, preprocessing pipeline, hyperparameters, training process and saved model outputs.

* **Model 2** — Documents the XGBoost implementation, preprocessing pipeline, hyperparameters, training process and saved model outputs.

---

## Performance and Statistical Documentation

The documentation describes the evaluation of the trained models and the statistical comparison between them.

* **Model 1 Performance** — Documents the performance evaluation of the Random Forest classifier using Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix analysis and bootstrap 95% confidence intervals.

* **Model 2 Performance** — Documents the performance evaluation of the XGBoost classifier using Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix analysis and bootstrap 95% confidence intervals.

* **Model Comparison** — Documents the comparison between the Random Forest and XGBoost models using side-by-side performance metrics, metric differences, paired bootstrap confidence intervals, McNemar's test, confusion matrices and ROC curves.

---

## Client Recommendation

* **Client Recommendation** — Provides the client recommendation report based on the results presented in Part C. The report discusses the model selected for the STADIOalot return prediction problem, potential improvements to the selected model, adaptation requirements for the requested SS1 data and whether the modelling results align with the related literature reviewed.

The report provides a formal business-oriented interpretation of the experimental findings and translates the technical results into recommendations relevant to the STADIOalot return prediction use case.

---

## Relationship to the Notebooks

The technical documentation corresponds to the project's Jupyter notebooks:

| Documentation           | Associated Notebook            |
| ----------------------- | ------------------------------ |
| `Preprocessing.MD`      | `01_Preprocessing.ipynb`       |
| `FeatureEngineering.MD` | `02_FeatureEngineering.ipynb`  |
| `Model1.MD`             | `03_Model1_RandomForest.ipynb` |
| `Model2.MD`             | `04_Model2_XGBoost.ipynb`      |
| `Model1Performance.MD`  | `05_Model1Performance.ipynb`   |
| `Model2Performance.MD`  | `06_Model2Performance.ipynb`      |
| `Comparison.MD`         | `07_ModelComparison.ipynb`        |

The notebooks contain the executable implementation, while the corresponding Markdown documents explain the methodology, implementation decisions, outputs, evaluation procedures and results.

The performance and comparison notebooks use the trained model pipelines saved during model development rather than retraining the models for every evaluation.

---

## Documentation Workflow

The relationship between the documentation and executable notebooks can be represented as follows:

```text
Data Request
     │
     ▼
Literature Review
     │
     ▼
01_Preprocessing.ipynb
     │
     ▼
Preprocessing.MD
     │
     ▼
02_FeatureEngineering.ipynb
     │
     ▼
FeatureEngineering.MD
     │
     ├─────────────────────────────┐
     ▼                             ▼
03_Model1_RandomForest.ipynb   04_Model2_XGBoost.ipynb
     │                             │
     ▼                             ▼
Model1.MD                       Model2.MD
     │                             │
     ▼                             ▼
05_Model1Performance.ipynb   06_Model2Performance.ipynb
     │                             │
     ▼                             ▼
Model1Performance.MD            Model2Performance.MD
     │                             │
     └──────────────┬──────────────┘
                    ▼
          07_ModelComparison.ipynb
                    │
                    ▼
              Comparison.MD
```

---

## Purpose

The documentation stored in this folder provides supporting information for the development, implementation, evaluation and comparison of the **STADIOalot Return Prediction** solution.

Together, these documents provide:

1. The data requirements and research background.
2. The preprocessing and data preparation methodology.
3. The feature engineering methodology.
4. The implementation of Model 1.
5. The implementation of Model 2.
6. The performance evaluation of Model 1.
7. The performance evaluation of Model 2.
8. The statistical comparison of Model 1 and Model 2.
9. A clear connection between the technical documentation and executable notebooks.
10. Supporting information for the final project report and evaluation.

The documentation is intended to make the project methodology, implementation, experimental results and statistical evaluation clear and reproducible.
