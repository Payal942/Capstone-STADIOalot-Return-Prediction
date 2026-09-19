# Visualisations

This folder contains the visual outputs generated during the **STADIOalot Return Prediction** project.

The visualisations are used to communicate and interpret the performance of the machine-learning models developed to predict whether an e-commerce order will be **Returned** or **Not Returned**.

## Contents

The folder currently contains the following model-performance visualisations:

| Visualisation                                | File                                        | Description                                                                                                                                 |
| -------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model 1 — Random Forest Confusion Matrix** | `model1_random_forest_confusion_matrix.png` | Shows the classification results of the Random Forest model, including true positives, true negatives, false positives and false negatives. |
| **Model 2 — XGBoost Confusion Matrix**       | `model2_xgboost_confusion_matrix.png`       | Shows the classification results of the XGBoost model, including true positives, true negatives, false positives and false negatives.       |

## Model Performance Visualisations

### Random Forest Confusion Matrix

**File:**

`visualisations/model1_random_forest_confusion_matrix.png`

The Random Forest confusion matrix provides a visual representation of the model's classification results on the test dataset.

It shows:

* **True Negatives (TN)** — orders correctly predicted as Not Returned
* **False Positives (FP)** — orders predicted as Returned when they were Not Returned
* **False Negatives (FN)** — orders predicted as Not Returned when they were Returned
* **True Positives (TP)** — orders correctly predicted as Returned

### XGBoost Confusion Matrix

**File:**

`visualisations/model2_xgboost_confusion_matrix.png`

The XGBoost confusion matrix provides the same classification breakdown for the XGBoost model on the test dataset.

It can be used alongside the model evaluation metrics to understand the types of prediction errors made by the model.

## Purpose of the Visualisations

The visualisations support the evaluation and interpretation of the machine-learning models by making the classification results easier to understand.

In particular, the confusion matrices help identify:

* Correct classifications
* Incorrect classifications
* False positives
* False negatives
* Differences in classification behaviour between the two models

These visual outputs complement the numerical evaluation results stored in the `experimental_results` folder.

## Related Project Files

The visualisations are generated as part of the model-training and evaluation process.

Related files include:

* **Model 1 notebook:** `notebooks/03_Model1_RandomForest.ipynb`
* **Model 2 notebook:** `notebooks/04_Model2_XGBoost.ipynb`
* **Model 1 trained model:** `models/model1_random_forest.joblib`
* **Model 2 trained model:** `models/model2_xgboost.joblib`
* **Model 1 metrics:** `experimental_results/model1_random_forest_metrics.csv`
* **Model 2 metrics:** `experimental_results/model2_xgboost_metrics.csv`

## Reproducibility

The confusion matrices are generated during model evaluation in the respective model-training notebooks.

Re-running the model notebooks with the required datasets and dependencies will regenerate the corresponding visualisation files.

## File Format

The generated visualisations are saved as **PNG (`.png`) image files** so that they can be viewed easily in GitHub, documentation and project presentations.
