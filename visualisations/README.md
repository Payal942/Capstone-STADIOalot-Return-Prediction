# Visualisations

This folder contains the visual outputs generated during the **STADIOalot Return Prediction** project.

The visualisations support the interpretation and evaluation of machine-learning models developed to predict whether an e-commerce order will be **Returned** or **Not Returned**. They include confusion matrices, training and validation loss plots, performance comparisons and ROC curves.

## Contents

The folder contains the following model-performance visualisations.

| Visualisation | File | Description |
|---|---|---|
| Model 1 — Random Forest Confusion Matrix | `model1_random_forest_confusion_matrix.png` | Shows the Random Forest model's classification results, including true positives, true negatives, false positives and false negatives. |
| Model 2 — XGBoost Confusion Matrix | `model2_xgboost_confusion_matrix.png` | Shows the XGBoost model's classification results, including true positives, true negatives, false positives and false negatives. |
| Visualisation 1 — Training and Validation Log Loss | `visualisation_1_training_validation_logloss.png` | Displays training and validation log loss across boosting iterations to help assess learning progress and potential overfitting. |
| Visualisation 2 — Performance Across Data Splits | `visualisation_2_performance_splits.png` | Compares the model's evaluation metrics across the training, validation and unseen test datasets. |
| Visualisation 3 — Confusion Matrix | `visualisation_3_confusion_matrix.png` | Presents the classification results for returned and non-returned orders, highlighting correct predictions and classification errors. |
| Visualisation 4 — Unseen-Test ROC Curve | `visualisation_4_test_roc.png` | Shows the Receiver Operating Characteristic (ROC) curve for the unseen test dataset, illustrating the trade-off between the true-positive rate and false-positive rate at different classification thresholds. |
| Visualisation 5 — ROC Curves Across Data Splits | `visualisation_5_roc_splits.png` | Compares ROC curves across the training, validation and unseen test datasets to assess discrimination performance across the data splits. |

## Model Performance Visualisations

### Random Forest Confusion Matrix

**File:** `visualisations/model1_random_forest_confusion_matrix.png`

The Random Forest confusion matrix provides a visual representation of the model's classification results on the test dataset.

It shows:

- **True Negatives (TN):** Orders correctly predicted as Not Returned.
- **False Positives (FP):** Orders incorrectly predicted as Returned.
- **False Negatives (FN):** Orders incorrectly predicted as Not Returned when they were returned.
- **True Positives (TP):** Orders correctly predicted as Returned.

### XGBoost Confusion Matrix

**File:** `visualisations/model2_xgboost_confusion_matrix.png`

The XGBoost confusion matrix provides the same classification breakdown for the XGBoost model on the test dataset.

It can be examined alongside the numerical evaluation metrics to understand the types of prediction errors made by the model.

## Client-Data Workflow Visualisations

The following five visualisations support the evaluation of the XGBoost return-prediction workflow using the supplied STADIOalot client datasets.

### Visualisation 1 — Training and Validation Log Loss

**File:** `visualisations/visualisation_1_training_validation_logloss.png`

This plot displays the training and validation log loss throughout model training. Log loss measures the difference between the predicted probabilities and the actual target values, with lower values generally indicating better probabilistic predictions.

Comparing the two curves helps assess whether the model is learning effectively and whether its performance on the validation dataset begins to deteriorate relative to its training performance.

### Visualisation 2 — Performance Across Data Splits

**File:** `visualisations/visualisation_2_performance_splits.png`

This visualisation compares the evaluation metrics calculated for the training, validation and unseen test datasets.

It supports an assessment of how consistently the model performs across the different data splits. Differences in performance may indicate overfitting or limited generalisation, although interpretation should consider the specific metrics and dataset characteristics.

### Visualisation 3 — Confusion Matrix

**File:** `visualisations/visualisation_3_confusion_matrix.png`

This visualisation presents the model's classification results for the returned and non-returned order classes.

The confusion matrix helps identify correctly classified orders and the two main types of classification error:

- **False positives:** Non-returned orders incorrectly classified as returned.
- **False negatives:** Returned orders incorrectly classified as non-returned.

These errors have different business implications. In particular, false negatives may represent returned orders that the model failed to flag as potentially at risk.

### Visualisation 4 — Unseen-Test ROC Curve

**File:** `visualisations/visualisation_4_test_roc.png`

The ROC curve evaluates the model's ability to distinguish between returned and non-returned orders on the unseen test dataset across different classification thresholds.

The curve plots the true-positive rate against the false-positive rate. It can be interpreted alongside the ROC-AUC score, which summarises the model's discrimination ability. Performance should be assessed using the actual curve and reported score rather than assumed from the visualisation alone.

### Visualisation 5 — ROC Curves Across Data Splits

**File:** `visualisations/visualisation_5_roc_splits.png`

This visualisation compares ROC curves for the training, validation and unseen test datasets.

It helps assess whether the model's ability to distinguish between returned and non-returned orders remains consistent across the data splits. A substantial difference between training and test performance may indicate that the model does not generalise equally well to unseen observations.

## Purpose of the Visualisations

Together, the visualisations support the evaluation and interpretation of model behaviour by illustrating:

- Training progress and validation loss.
- Differences in performance across data splits.
- Correct classifications and classification errors.
- The model's ability to distinguish returned from non-returned orders.
- Potential differences between training performance and generalisation to unseen data.

The visualisations complement the numerical evaluation results stored in the `experimental_results/` folder. They do not replace the need to interpret the reported metrics and acknowledge model limitations.

## Related Project Files

The visualisations are generated as part of the model-training and evaluation process.

**Original model workflow**

- **Model 1 notebook:** `notebooks/03_Model1_RandomForest.ipynb`
- **Model 2 notebook:** `notebooks/04_Model2_XGBoost.ipynb`
- **Model 1 trained model:** `models/model1_random_forest.joblib`
- **Model 2 trained model:** `models/model2_xgboost.joblib`
- **Model 1 metrics:** `experimental_results/model1_random_forest_metrics.csv`
- **Model 2 metrics:** `experimental_results/model2_xgboost_metrics.csv`

**Client-data workflow**

- **Notebook:** `notebooks/SS3_Python.ipynb`
- **Trained XGBoost model:** `models/ss3_xgboost_return_prediction.joblib`
- **Fitted preprocessor:** `models/ss3_preprocessor.joblib`
- **Evaluation metrics:** `experimental_results/ss3_xgboost_metrics.csv`
- **Confusion-matrix results:** `experimental_results/ss3_xgboost_confusion_matrix.csv`
- **Training history:** `experimental_results/ss3_xgboost_training_history.csv`

## Reproducibility

The visualisations are generated during model training and evaluation in the relevant notebooks.

To regenerate them:

1. Ensure that the required datasets are available in the expected project directories.
2. Install the dependencies specified in `experimental_setup/requirements.txt`.
3. Open and execute the relevant notebook in the required order.
4. Confirm that the visualisation files are saved in the `visualisations/` directory.

The generated plots depend on the dataset, preprocessing steps, model configuration and evaluation procedures used during execution.

## File Format

The generated visualisations are saved as **PNG (`.png`) image files** so that they can be viewed easily in GitHub, documentation and project presentations.