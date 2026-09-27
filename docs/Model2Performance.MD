# Model 2 Performance — XGBoost Classifier

## 1. Overview

This document presents the performance analysis of **Model 2**, which uses an **XGBoost Classifier** to predict whether an e-commerce order will be returned.

The model was developed for the **STADIOalot Return Prediction** project. The purpose of this performance analysis is to evaluate Model 2 using the feature-engineered dataset and the saved XGBoost model produced during the model development stage.

The performance analysis includes:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Classification Report
* Confusion Matrix Summary
* Bootstrap 95% Confidence Intervals

The performance analysis is performed using the saved Model 2 pipeline rather than retraining the XGBoost model.

---

## 2. Model 2 Details

| Item                    | Description                           |
| ----------------------- | ------------------------------------- |
| Model                   | Model 2                               |
| Algorithm               | XGBoost Classifier                    |
| Problem Type            | Binary Classification                 |
| Target Variable         | `Return_Status`                       |
| Class `0`               | Not Returned                          |
| Class `1`               | Returned                              |
| Dataset                 | Feature-engineered STADIOalot dataset |
| Train/Test Split        | 80% training and 20% testing          |
| Random State            | `42`                                  |
| Model Training Notebook | `04_Model2_XGBoost.ipynb`             |
| Performance Notebook    | `06_Model2_Performance.ipynb`         |

The XGBoost model is trained in `04_Model2_XGBoost.ipynb`.

The trained pipeline is saved and subsequently loaded by `06_Model2_Performance.ipynb` for performance evaluation.

The performance notebook recreates the stratified 80/20 train/test split using the same `random_state=42` configuration used during model development.

This allows Model 2 to be evaluated using a consistent test-set methodology.

---

## 3. Input Dataset

The performance analysis uses the feature-engineered dataset generated during the data preparation and feature engineering stages.

```text
data/processed/returns_features.csv
```

The dataset contains the predictor variables and the target variable:

```text
Return_Status
```

The target variable is encoded as follows:

| Value | Meaning      |
| ----: | ------------ |
|   `0` | Not Returned |
|   `1` | Returned     |

The target column is separated from the predictor variables before the train/test split is recreated.

The performance notebook validates that:

* The dataset exists.
* The target variable is present.
* The target variable does not contain missing values.
* The target variable contains the expected binary classes.

---

## 4. Trained Model

The saved XGBoost pipeline is loaded from:

```text
models/model2_xgboost.joblib
```

This model is produced by:

```text
04_Model2_XGBoost.ipynb
```

The saved pipeline contains the trained preprocessing and XGBoost modelling components required to generate predictions.

The performance notebook verifies that the loaded pipeline contains:

* A preprocessing component.
* A trained model component.
* The `predict()` method.
* The `predict_proba()` method.

These methods are required to generate class predictions and predicted return probabilities.

The performance notebook does **not** retrain the XGBoost model.

---

## 5. Performance Analysis Notebook

The Model 2 performance analysis is implemented in:

```text
06_Model2_Performance.ipynb
```

The notebook is responsible for:

1. Loading the feature-engineered dataset.
2. Validating the target variable.
3. Separating the predictor variables and target.
4. Recreating the stratified 80/20 train/test split.
5. Verifying the target distribution after splitting.
6. Loading the saved XGBoost pipeline.
7. Generating class predictions.
8. Generating return probabilities.
9. Calculating performance metrics.
10. Producing a classification report.
11. Producing a confusion-matrix summary.
12. Performing bootstrap resampling.
13. Calculating 95% bootstrap confidence intervals.
14. Saving the bootstrap confidence-interval results.

The notebook therefore focuses on **Model 2 evaluation and statistical analysis**, rather than model training.

---

## 6. How to Run the Notebook

The performance notebook should be run after the Model 2 training notebook has successfully created the trained XGBoost pipeline.

### Step 1: Install the Required Libraries

The project's dependency file is located in the `experimental_setup` folder.

From the project root directory, run:

```bash
pip install -r experimental_setup/requirements.txt
```

### Step 2: Open the Performance Notebook

Open the following notebook:

```text
06_Model2_Performance.ipynb
```

The notebook can be opened using Jupyter Notebook or the Jupyter extension in Visual Studio Code.

For example:

```bash
jupyter notebook
```

### Step 3: Ensure the Required Files Exist

Before running the performance notebook, the following files must exist:

```text
data/processed/returns_features.csv
models/model2_xgboost.joblib
```

The XGBoost model file should have been created by:

```text
04_Model2_XGBoost.ipynb
```

### Step 4: Execute the Notebook

Run the cells in:

```text
06_Model2_Performance.ipynb
```

from top to bottom.

The notebook recreates the train/test split using:

```text
test_size = 0.20
random_state = 42
stratify = y
```

It then evaluates the saved XGBoost pipeline on the recreated test set.

---

## 7. Recreating the Test Set

The performance notebook recreates the Model 2 train/test split using a stratified 80/20 split.

The split configuration is:

| Parameter           | Value |
| ------------------- | ----: |
| Training proportion |   80% |
| Testing proportion  |   20% |
| Random State        |  `42` |
| Stratification      |   Yes |

The split is performed using:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Stratification maintains the target-class distribution as closely as possible between the training and testing sets.

The recreated test set is used to generate:

* Predicted classes
* Return probabilities
* Performance metrics
* Classification report
* Confusion-matrix summary
* Bootstrap confidence intervals

Using the same split configuration as Model 1 supports a consistent evaluation methodology when comparing the two models.

---

## 8. Prediction Outputs

The loaded XGBoost pipeline produces two main prediction outputs.

### 8.1 Class Predictions

Class predictions are generated using:

```python
y_pred = model_pipeline.predict(X_test)
```

The predicted values are:

| Prediction | Meaning                |
| ---------: | ---------------------- |
|        `0` | Predicted Not Returned |
|        `1` | Predicted Returned     |

These predictions are used to calculate Accuracy, Precision, Recall and F1-Score.

### 8.2 Return Probabilities

The probability of an order being returned is generated using:

```python
y_probability = model_pipeline.predict_proba(X_test)[:, 1]
```

This represents the model's estimated probability that an order belongs to the `Returned` class.

The predicted probabilities are used to calculate ROC-AUC.

The notebook also checks that the number of class predictions and probability predictions matches the number of observations in the test set.

---

## 9. Performance Metrics

The performance notebook calculates five primary performance metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

### 9.1 Accuracy

Accuracy measures the overall proportion of correct predictions.

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

Accuracy provides an overall measure of prediction correctness.

It should be interpreted together with the other performance measures because it does not distinguish between different types of classification errors.

---

### 9.2 Precision

Precision measures the proportion of orders predicted as returned that were actually returned.

$$
Precision = \frac{TP}{TP + FP}
$$

Precision provides information about the reliability of predictions for the `Returned` class.

A false positive occurs when an order is predicted as returned but was actually not returned.

---

### 9.3 Recall

Recall measures the proportion of actual returned orders that were correctly identified.

$$
Recall = \frac{TP}{TP + FN}
$$

Recall provides information about how many of the actual returned orders were detected by the model.

A false negative occurs when an order was actually returned but the model predicted that it would not be returned.

---

### 9.4 F1-Score

The F1-Score is the harmonic mean of Precision and Recall.

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

The F1-Score provides a combined measure of Precision and Recall.

It is useful when both types of performance need to be considered together.

---

### 9.5 ROC-AUC

ROC-AUC measures the model's ability to distinguish between returned and non-returned orders across different classification thresholds.

ROC-AUC is calculated using the predicted return probabilities rather than the predicted class labels.

A ROC-AUC value closer to 1 indicates stronger separation between the two target classes, while a value around 0.5 indicates performance close to random ranking.

---

### 9.6 Confusion Matrix

The confusion matrix summarises the correct and incorrect predictions made by the model.

| Measure             | Description                                          |
| ------------------- | ---------------------------------------------------- |
| True Negative (TN)  | Correctly predicted non-returned order               |
| False Positive (FP) | Non-returned order incorrectly predicted as returned |
| False Negative (FN) | Returned order incorrectly predicted as non-returned |
| True Positive (TP)  | Correctly predicted returned order                   |

The notebook calculates the four confusion-matrix values:

* True Negatives (TN)
* False Positives (FP)
* False Negatives (FN)
* True Positives (TP)

The confusion matrix is presented as a **summary table** in the notebook. No separate confusion-matrix image file is produced by the performance notebook.

---

## 10. Classification Report

The performance notebook generates a classification report using the test-set predictions.

The report provides class-level performance information for:

* Not Returned
* Returned
* Macro Average
* Weighted Average

The classification report contains:

* Precision
* Recall
* F1-Score
* Support

The classification report provides additional detail beyond the five overall performance metrics by showing performance separately for each target class.

The report is displayed during notebook execution and is not saved as a separate output file.

---

## 11. Bootstrap 95% Confidence Intervals

Bootstrap resampling is used to estimate uncertainty around the observed performance metrics.

The Model 2 performance notebook uses the following configuration:

| Parameter                      |                     Value |
| ------------------------------ | ------------------------: |
| Bootstrap iterations requested |                   `1,000` |
| Confidence level               |                     `95%` |
| Bootstrap random state         |                      `42` |
| Resampling method              | Sampling with replacement |
| Resampling unit                |     Test-set observations |

For each bootstrap iteration, the notebook samples observations from the test set with replacement and recalculates:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

The 95% confidence interval is calculated using the 2.5th and 97.5th percentiles of the bootstrap results.

For example:

$$
CI_{95\%} =
[P_{2.5}, P_{97.5}]
$$

where:

* $P_{2.5}$ is the 2.5th percentile.
* $P_{97.5}$ is the 97.5th percentile.

### ROC-AUC Bootstrap Handling

ROC-AUC requires both target classes to be present in the bootstrap sample.

If a bootstrap sample contains only one target class, ROC-AUC cannot be calculated for that iteration.

That bootstrap iteration is therefore excluded from the ROC-AUC bootstrap results.

The number of valid bootstrap samples is recorded separately for each metric in the output CSV.

---

## 12. Statistical Output

The bootstrap confidence-interval results are saved to:

```text
experimental_results/model2_bootstrap_confidence_intervals.csv
```

The output contains the following columns:

| Column                           | Description                              |
| -------------------------------- | ---------------------------------------- |
| `Model`                          | Model identifier                         |
| `Algorithm`                      | Algorithm used by the model              |
| `Metric`                         | Performance metric                       |
| `Observed Score`                 | Original test-set performance score      |
| `95% CI Lower`                   | Lower confidence-interval boundary       |
| `95% CI Upper`                   | Upper confidence-interval boundary       |
| `Bootstrap Samples`              | Number of valid bootstrap samples        |
| `Bootstrap Iterations Requested` | Requested number of bootstrap iterations |
| `Random State`                   | Bootstrap random state                   |

The five primary metrics are represented in the bootstrap output:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

The bootstrap output is the statistical output generated by the Model 2 performance notebook.

---

## 13. Existing Model 2 Metrics

The original Model 2 performance metrics are stored separately in:

```text
experimental_results/model2_xgboost_metrics.csv
```

This metrics file is associated with the Model 2 training and evaluation process in:

```text
04_Model2_XGBoost.ipynb
```

The performance notebook recalculates the metrics from the recreated test set for evaluation and uses those observed scores as the basis for the bootstrap analysis.

The performance notebook does **not** overwrite the original Model 2 metrics file.

The separate bootstrap file is created for the Part C statistical analysis.

---

## 14. Output Files

The Model 2 performance analysis uses and produces the following files:

| File                                                             | Purpose                                                                  |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `data/processed/returns_features.csv`                            | Input feature-engineered dataset                                         |
| `models/model2_xgboost.joblib`                                   | Saved trained XGBoost pipeline                                           |
| `experimental_results/model2_xgboost_metrics.csv`                | Existing Model 2 performance metrics                                     |
| `experimental_results/model2_bootstrap_confidence_intervals.csv` | Bootstrap 95% confidence intervals generated by the performance notebook |

### File Generation Summary

| File                                        | Used by `06_Model2_Performance.ipynb` | Created/Updated by `06_Model2_Performance.ipynb` |
| ------------------------------------------- | ------------------------------------- | ------------------------------------------------ |
| `returns_features.csv`                      | Yes                                   | No                                               |
| `model2_xgboost.joblib`                     | Yes                                   | No                                               |
| `model2_xgboost_metrics.csv`                | Existing Model 2 result               | No                                               |
| `model2_bootstrap_confidence_intervals.csv` | No                                    | **Yes**                                          |

Therefore, `06_Model2_Performance.ipynb` does not retrain the model or overwrite the existing model metrics.

---

## 15. Validation Checks

The performance notebook performs validation checks to confirm that the evaluation can be completed correctly.

These checks include:

* The input dataset exists.
* The target variable is present.
* The target variable contains no missing values.
* The target variable contains the expected binary classes.
* The saved Model 2 file exists.
* The model loads successfully.
* The loaded pipeline contains a preprocessing component.
* The loaded pipeline contains a model component.
* The model has a `predict()` method.
* The model has a `predict_proba()` method.
* The number of class predictions matches the number of test observations.
* The number of probability predictions matches the number of test observations.
* The bootstrap output file is successfully saved.

These checks help identify missing files, invalid model files and prediction-size mismatches before the statistical results are finalised.

---

## 16. Model 2 Performance Workflow

The Model 2 performance workflow is:

```text
Feature-engineered dataset
            |
            v
Separate predictors and target
            |
            v
Recreate stratified 80/20 train/test split
            |
            v
Load saved XGBoost pipeline
            |
            v
Generate class predictions
            |
            v
Generate return probabilities
            |
            v
Calculate performance metrics
            |
            v
Generate classification report
            |
            v
Generate confusion-matrix summary
            |
            v
Perform 1,000 bootstrap iterations
            |
            v
Calculate 95% confidence intervals
            |
            v
Save bootstrap statistical results
```

The workflow uses the trained XGBoost pipeline created by:

```text
04_Model2_XGBoost.ipynb
```

The performance analysis is performed by:

```text
06_Model2_Performance.ipynb
```

No model retraining takes place during the performance analysis.

---

## 17. Relationship Between Model 2 Training and Performance

The Model 2 workflow consists of two separate stages.

### Model Development

```text
04_Model2_XGBoost.ipynb
        |
        v
Train XGBoost model
        |
        v
Save trained pipeline
        |
        v
models/model2_xgboost.joblib
```

### Model Performance

```text
06_Model2_Performance.ipynb
        |
        v
Load trained pipeline
        |
        v
Recreate test set
        |
        v
Generate predictions
        |
        v
Calculate performance metrics
        |
        v
Perform bootstrap analysis
        |
        v
experimental_results/model2_bootstrap_confidence_intervals.csv
```

This separation ensures that model training and statistical performance evaluation are documented as distinct stages of the project.

---

## 18. Summary

Model 2 performance is evaluated using the test portion of the feature-engineered STADIOalot return-prediction dataset.

Model 2 uses an **XGBoost Classifier** and predicts:

```text
0 = Not Returned
1 = Returned
```

The performance analysis includes:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Classification Report
* Confusion-Matrix Summary
* Bootstrap 95% Confidence Intervals

The test set is recreated using:

```text
80% training
20% testing
random_state = 42
stratification = y
```

The statistical analysis uses:

* 1,000 bootstrap iterations
* Sampling with replacement
* Test-set observations as the resampling unit
* A 95% confidence level
* Bootstrap random state `42`
* Percentile-based confidence intervals

The Model 2 training notebook is:

```text
04_Model2_XGBoost.ipynb
```

The Model 2 performance notebook is:

```text
06_Model2_Performance.ipynb
```

The trained XGBoost pipeline is:

```text
models/model2_xgboost.joblib
```

The existing Model 2 metrics file is:

```text
experimental_results/model2_xgboost_metrics.csv
```

The statistical output generated by the performance notebook is:

```text
experimental_results/model2_bootstrap_confidence_intervals.csv
```

Overall, `06_Model2_Performance.ipynb` provides the performance and statistical evaluation of the trained XGBoost model without retraining or modifying the saved Model 2 pipeline.
