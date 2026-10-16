# Visualisations

This folder contains the visual outputs generated during the **STADIOalot Return Prediction** project.

The visualisations support the interpretation and evaluation of machine-learning models developed to predict whether an e-commerce order will be **Returned** or **Not Returned**. They include confusion matrices, training and validation loss plots, performance comparisons, ROC curves, class-imbalance analysis, missing-value analysis, feature importance, monthly performance, and target-leakage experiments.

## Contents

The folder contains the following model-performance and client-data visualisations.

| Visualisation | File | Description |
|---|---|---|
| Model 1 — Random Forest Confusion Matrix | `model1_random_forest_confusion_matrix.png` | Shows the Random Forest model's classification results, including true positives, true negatives, false positives and false negatives. |
| Model 2 — XGBoost Confusion Matrix | `model2_xgboost_confusion_matrix.png` | Shows the XGBoost model's classification results, including true positives, true negatives, false positives and false negatives. |
| Visualisation 1 — Training and Validation Log Loss | `visualisation_1_training_validation_logloss.png` | Displays training and validation log loss across boosting iterations to help assess learning progress and potential overfitting. |
| Visualisation 2 — Performance Across Data Splits | `visualisation_2_performance_splits.png` | Compares model evaluation metrics across the training, validation and unseen-test datasets. |
| Visualisation 3 — Unseen-Test Confusion Matrix | `visualisation_3_confusion_matrix.png` | Shows correct classifications and classification errors on the unseen test dataset. |
| Visualisation 4 — Unseen-Test ROC Curve | `visualisation_4_test_roc.png` | Illustrates the trade-off between the true-positive rate and false-positive rate across classification thresholds. |
| Visualisation 5 — ROC Curves Across Data Splits | `visualisation_5_roc_splits.png` | Compares ROC curves across training, validation and unseen-test datasets. |
| Visualisation 6 — Return Class Imbalance | `visualisation_6_class_imbalance.png` | Shows the distribution of returned and non-returned orders in the client dataset. |
| Visualisation 7 — Missing Values Across Client Tables | `visualisation_7_missing_values.png` | Summarises missing values across the supplied STADIOalot client tables. |
| Visualisation 8 — XGBoost Feature Importance | `visualisation_8_feature_importance.png` | Displays the most influential features according to the fitted XGBoost model's feature-importance measure. |
| Visualisation 9 — Performance by Order Month | `visualisation_9_performance_by_month.png` | Examines model performance across order months to identify potential variation over time. |
| Visualisation 10 — Target Leakage Analysis | `visualisation_10_target_leakage_analysis.png` | Examines return rates across delivery-related categories, including on-time status, delivery attempts and actual delivery timing. |
| Visualisation 11A — Leakage Variables and Returns | `visualisation_11a_leakage_vs_returns.png` | Illustrates the relationship between delivery-related variables and return outcomes. |
| Visualisation 11B — Model Performance With vs Without Leakage | `visualisation_11b_model_performance_comparison.png` | Compares model metrics when delivery-related variables are excluded from or included in separate experimental feature sets. |

## Model Performance Visualisations

### Random Forest Confusion Matrix

**File:** `visualisations/model1_random_forest_confusion_matrix.png`

The Random Forest confusion matrix provides a visual representation of the model's classification results on its evaluation dataset.

It shows:

- **True Negatives (TN):** Orders correctly predicted as Not Returned.
- **False Positives (FP):** Orders incorrectly predicted as Returned.
- **False Negatives (FN):** Returned orders incorrectly predicted as Not Returned.
- **True Positives (TP):** Orders correctly predicted as Returned.

These results help explain the types of classification errors made by the Random Forest model.

### XGBoost Confusion Matrix

**File:** `visualisations/model2_xgboost_confusion_matrix.png`

The XGBoost confusion matrix provides the equivalent classification breakdown for the XGBoost model.

It can be examined alongside the numerical evaluation metrics to understand the model's ability to identify returned orders and the number of returns it misses.

## Client-Data Workflow Visualisations

The following visualisations support the evaluation of the XGBoost return-prediction workflow using the supplied STADIOalot client datasets.

### Visualisation 1 — Training and Validation Log Loss

**File:** `visualisations/visualisation_1_training_validation_logloss.png`

This plot displays training and validation log loss throughout model training. Log loss measures the quality of predicted probabilities, with lower values generally indicating better probabilistic predictions.

Comparing the two curves helps assess learning progress and identify potential overfitting when validation performance deteriorates relative to training performance.

### Visualisation 2 — Performance Across Data Splits

**File:** `visualisations/visualisation_2_performance_splits.png`

This visualisation compares the evaluation metrics calculated for the training, validation and unseen-test datasets.

It supports an assessment of model consistency across data splits. Differences may indicate overfitting or limited generalisation and should be interpreted in the context of the individual metrics and dataset characteristics.

### Visualisation 3 — Unseen-Test Confusion Matrix

**File:** `visualisations/visualisation_3_confusion_matrix.png`

This visualisation presents the model's classification results for returned and non-returned orders in the unseen-test dataset.

The confusion matrix distinguishes correct predictions from two important types of error:

- **False positives:** Non-returned orders incorrectly classified as returned.
- **False negatives:** Returned orders incorrectly classified as non-returned.

False negatives are particularly important for this business problem because they represent returns the model failed to flag as potentially at risk.

### Visualisation 4 — Unseen-Test ROC Curve

**File:** `visualisations/visualisation_4_test_roc.png`

The Receiver Operating Characteristic (ROC) curve evaluates the model's ability to distinguish between returned and non-returned orders across different classification thresholds.

It plots the true-positive rate against the false-positive rate and should be interpreted alongside the ROC-AUC score. A higher ROC-AUC generally indicates better discrimination between the two classes.

### Visualisation 5 — ROC Curves Across Data Splits

**File:** `visualisations/visualisation_5_roc_splits.png`

This visualisation compares ROC curves for the training, validation and unseen-test datasets.

It helps assess whether the model's discrimination ability remains consistent across data splits. A substantial difference between training and unseen-test performance may indicate that the model does not generalise equally well to new observations.

### Visualisation 6 — Return Class Imbalance

**File:** `visualisations/visualisation_6_class_imbalance.png`

This graph shows the distribution of returned and non-returned orders in the client dataset.

Class imbalance matters because a model can achieve relatively high accuracy by favouring the majority class while failing to identify a substantial proportion of returned orders. Precision, recall, F1-score, ROC-AUC and the confusion matrix should therefore be considered alongside accuracy.

### Visualisation 7 — Missing Values Across Client Tables

**File:** `visualisations/visualisation_7_missing_values.png`

This visualisation summarises missing-value patterns across the supplied client tables.

It supports data-quality assessment and helps identify where missing information may require preprocessing. Missing-value handling should be selected according to the variable's meaning, the extent of missingness and the modelling requirements.

### Visualisation 8 — XGBoost Feature Importance

**File:** `visualisations/visualisation_8_feature_importance.png`

This graph displays the leading features according to the fitted XGBoost model's feature-importance measure.

It helps identify variables that contributed to the model's predictions. Feature importance indicates predictive contribution under the selected importance method; it does not establish causation or necessarily indicate the direction of a feature's relationship with returns.

### Visualisation 9 — Performance by Order Month

**File:** `visualisations/visualisation_9_performance_by_month.png`

This visualisation examines model performance across order months.

Monthly comparisons can reveal variation in accuracy and recall that may be hidden by aggregate metrics. Results for months with relatively few observations should be interpreted cautiously because their metrics may be less stable.

### Visualisation 10 — Target Leakage Analysis

**File:** `visualisations/visualisation_10_target_leakage_analysis.png`

This visualisation examines observed return rates across delivery-related categories, including on-time status, delivery attempts and actual delivery timing.

The analysis provides descriptive evidence about the relationship between delivery information and return outcomes. Differences in return rates should not be interpreted as evidence that delivery conditions cause returns.

The timing of these variables is particularly important. Delivery information is generally recorded after an order has been placed and may not be available when a prediction is required at order confirmation. Such variables are therefore unsuitable for the intended prediction point, even if they appear useful during exploratory analysis.

### Visualisation 11A — Leakage Variables and Returns

**File:** `visualisations/visualisation_11a_leakage_vs_returns.png`

This visualisation further examines the relationship between delivery-related variables and return status.

It supports the investigation of whether post-purchase information is associated with the target. These relationships should be considered in conjunction with the prediction timeline to distinguish genuinely available predictive features from information that becomes available only after the order has progressed.

### Visualisation 11B — Model Performance With vs Without Leakage

**File:** `visualisations/visualisation_11b_model_performance_comparison.png`

This visualisation compares experimental XGBoost results from a baseline feature set against separate experiments that include individual delivery-related variables or all three variables together.

The comparison considers ROC-AUC, PR-AUC, precision, recall and F1-score. In the recorded experiment, including the delivery-related variables did not improve F1-score over the baseline. Some metrics changed slightly between experiments, but these differences do not make post-purchase variables appropriate for an order-confirmation prediction system.

These experiments use a separate 80/20 evaluation split and should not be confused with the main SS3 model's training, validation and unseen-test results. The main model's reported evaluation metrics must remain separate from the leakage experiment metrics.

## Purpose of the Visualisations

Together, the visualisations support the evaluation and interpretation of the return-prediction workflow by illustrating:

- Training progress and validation loss.
- Performance differences across data splits.
- Correct classifications and classification errors.
- The model's ability to distinguish returned from non-returned orders.
- Class imbalance and data-quality considerations.
- Features identified as important by the fitted model.
- Variation in performance across order months.
- Potential target-leakage risks associated with post-purchase information.
- Experimental performance comparisons with and without delivery-related variables.

The visualisations complement the numerical evaluation results stored in the `experimental_results/` directory. They do not replace the need to interpret reported metrics, consider class imbalance, assess generalisation and acknowledge model limitations.

## Related Project Files

The visualisations are generated as part of the model-training and evaluation workflow.

### Original Model Workflow

- **Model 1 notebook:** `notebooks/03_Model1_RandomForest.ipynb`
- **Model 2 notebook:** `notebooks/04_Model2_XGBoost.ipynb`
- **Model 1 trained model:** `models/model1_random_forest.joblib`
- **Model 2 trained model:** `models/model2_xgboost.joblib`
- **Model 1 metrics:** `experimental_results/model1_random_forest_metrics.csv`
- **Model 2 metrics:** `experimental_results/model2_xgboost_metrics.csv`

### Client-Data Workflow

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
3. Open and execute the relevant notebook cells in the required order.
4. Confirm that the generated PNG files are saved in the `visualisations/` directory.

The generated plots depend on the dataset, preprocessing steps, model configuration and evaluation procedures used during execution.

## File Format

The generated visualisations are saved as **PNG (`.png`) image files** so they can be viewed easily in GitHub, documentation and project presentations.