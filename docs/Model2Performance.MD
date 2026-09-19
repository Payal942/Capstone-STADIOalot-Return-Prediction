# Model 2 Performance — XGBoost Classifier

## 1. Overview

This document presents the performance results of **Model 2**, which uses an **XGBoost Classifier** to predict whether an e-commerce order will be returned.

The model was developed for the **STADIOalot Return Prediction** project. The purpose of this performance analysis is to evaluate how accurately Model 2 identifies returned and non-returned orders using the public dataset selected for the project.

The performance analysis includes:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Classification report
* Confusion matrix
* Bootstrap 95% confidence intervals

---

## 2. Model 2 Details

| Item | Description |
| ------------------- | ------------------------------------- |
| Model | Model 2 |
| Algorithm | XGBoost Classifier |
| Problem Type | Binary classification |
| Target Variable | `Return_Status` |
| Class `0` | Not Returned |
| Class `1` | Returned |
| Dataset | Feature-engineered STADIOalot dataset |
| Train/Test Split | 80% training and 20% testing |
| Random State | `42` |
| Evaluation Notebook | `Model2Performance.ipynb` |

The performance notebook loads the trained XGBoost pipeline produced during Part B and evaluates it on the recreated test set.

The same stratified train/test split configuration used for Model 1 is applied to Model 2 to ensure that the two models are evaluated using the same test-set methodology.

---

## 3. Input Dataset

The performance analysis uses the feature-engineered dataset generated during the data preparation and feature engineering stages.

```text
data/processed/returns_features.csv
````

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

---

## 4. Trained Model

The saved XGBoost pipeline is loaded from the following file:

```text
models/model2_xgboost.joblib
```

The saved pipeline contains the trained preprocessing and XGBoost modelling components required to generate predictions.

The notebook verifies that the loaded pipeline contains:

* A preprocessing component.
* A trained model component.
* The `predict()` method.
* The `predict_proba()` method.

These methods are required to generate class predictions and return probabilities.

---

## 5. Performance Analysis Notebook

The Model 2 performance calculations are implemented in:

```text
Model2Performance.ipynb
```

This notebook is responsible for:

1. Loading the feature-engineered dataset.
2. Validating the target variable.
3. Recreating the Model 2 test set.
4. Loading the saved XGBoost pipeline.
5. Generating class predictions.
6. Generating return probabilities.
7. Calculating performance metrics.
8. Producing a classification report.
9. Creating a confusion matrix summary.
10. Calculating bootstrap 95% confidence intervals.
11. Saving the statistical results.

---

## 6. How to Run the Notebook

Run the notebook from the root directory of the project.

### Step 1: Install the required libraries

```bash
pip install -r requirements.txt
```

### Step 2: Open the notebook

The notebook can be opened using Jupyter Notebook or the Jupyter extension in Visual Studio Code.

```bash
jupyter notebook
```

Open:

```text
notebooks/Model2Performance.ipynb
```

### Step 3: Execute the notebook

Run the notebook cells from top to bottom.

The notebook requires the following files to exist before execution:

```text
data/processed/returns_features.csv
models/model2_xgboost.joblib
```

The notebook will recreate the 80/20 stratified test split using:

```text
test_size = 0.20
random_state = 42
stratify = y
```

The notebook then generates Model 2 predictions, performance metrics, classification results, confusion matrix information, and bootstrap confidence intervals.

---

## 7. Recreating the Test Set

To ensure that the performance analysis evaluates the same type of test data used during model development, the notebook recreates the train/test split using a stratified 80/20 split.

The split configuration is:

| Parameter           | Value |
| ------------------- | ----: |
| Training proportion |   80% |
| Testing proportion  |   20% |
| Random state        |  `42` |
| Stratification      |   Yes |

Stratification ensures that the class distribution of `Return_Status` is maintained as closely as possible in both the training and testing sets.

The test set is used to generate:

* Predicted classes
* Return probabilities
* Evaluation metrics
* Classification report
* Confusion matrix
* Bootstrap confidence intervals

Using the same split configuration as Model 1 supports a consistent comparison between the two models.

---

## 8. Prediction Outputs

The loaded XGBoost model produces two types of predictions.

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

### 8.2 Return Probabilities

The probability of an order being returned is generated using:

```python
y_probability = model_pipeline.predict_proba(X_test)[:, 1]
```

This probability represents the model's estimated likelihood that an order belongs to the `Returned` class.

The probability predictions are used for calculating ROC-AUC.

---

## 9. Performance Metrics

The notebook calculates the following performance metrics.

### 9.1 Accuracy

Accuracy measures the overall proportion of correct predictions.

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

Accuracy is useful for measuring overall correctness, but it should be interpreted together with precision, recall, F1-score, and ROC-AUC.

---

### 9.2 Precision

Precision measures the proportion of orders predicted as returned that were actually returned.

$$
Precision = \frac{TP}{TP + FP}
$$

A higher precision means that fewer orders are incorrectly identified as returned.

For the STADIOalot return-prediction problem, precision provides information about the reliability of predicted returns.

---

### 9.3 Recall

Recall measures the proportion of actual returned orders that were correctly identified by the model.

$$
Recall = \frac{TP}{TP + FN}
$$

Recall is important because identifying actual returned orders is a central objective of the predictive system.

---

### 9.4 F1-Score

The F1-score is the harmonic mean of precision and recall.

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

The F1-score provides a combined measure of precision and recall.

---

### 9.5 ROC-AUC

ROC-AUC measures the model's ability to distinguish between returned and non-returned orders across different classification thresholds.

The ROC-AUC score is calculated using the predicted return probabilities rather than the predicted class labels.

A higher ROC-AUC indicates stronger discrimination between the two target classes.

---

### 9.6 Confusion Matrix

The confusion matrix summarises the correct and incorrect predictions made by the model.

| Measure        | Description                                          |
| -------------- | ---------------------------------------------------- |
| True Negative  | Correctly predicted non-returned order               |
| False Positive | Non-returned order incorrectly predicted as returned |
| False Negative | Returned order incorrectly predicted as non-returned |
| True Positive  | Correctly predicted returned order                   |

The notebook extracts the following four values from the confusion matrix:

* True Negatives (TN)
* False Positives (FP)
* False Negatives (FN)
* True Positives (TP)

These values provide additional information about the types of prediction errors made by Model 2.

---

## 10. Classification Report

The notebook generates a classification report containing class-level performance measures for:

* Not Returned
* Returned
* Macro Average
* Weighted Average

The classification report includes:

* Precision
* Recall
* F1-score
* Support

The report provides a more detailed view of the model's performance for each target class.

The `Returned` class is particularly relevant because identifying returned orders is the main prediction objective of the project.

---

## 11. Bootstrap Confidence Intervals

Bootstrap resampling is used to estimate uncertainty around the observed performance metrics.

The notebook uses:

| Parameter                      |                     Value |
| ------------------------------ | ------------------------: |
| Bootstrap iterations requested |                   `1,000` |
| Confidence level               |                     `95%` |
| Bootstrap random state         |                      `42` |
| Resampling method              | Sampling with replacement |
| Resampling unit                |     Test-set observations |

For each bootstrap iteration, the notebook resamples the test-set observations with replacement and recalculates:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

The 95% confidence interval is calculated using the 2.5th and 97.5th percentiles of the bootstrap results.

If a bootstrap sample contains only one target class, ROC-AUC cannot be calculated for that sample. Such a sample is therefore excluded from the ROC-AUC bootstrap results.

The confidence interval provides an indication of the uncertainty associated with each observed performance metric.

---

## 12. Statistical Output

The bootstrap confidence interval results are saved in a CSV file.

```text
experimental_results/model2_bootstrap_confidence_intervals.csv
```

The output contains the following information:

| Column                           | Description                              |
| -------------------------------- | ---------------------------------------- |
| `Model`                          | Model identifier                         |
| `Algorithm`                      | Algorithm used by the model              |
| `Metric`                         | Performance metric                       |
| `Observed Score`                 | Original test-set metric                 |
| `95% CI Lower`                   | Lower confidence interval boundary       |
| `95% CI Upper`                   | Upper confidence interval boundary       |
| `Bootstrap Samples`              | Number of valid bootstrap samples        |
| `Bootstrap Iterations Requested` | Requested number of bootstrap iterations |
| `Random State`                   | Bootstrap random state                   |

The Model 2 performance metrics generated during Part B are stored separately in:

```text
experimental_results/model2_xgboost_metrics.csv
```

The Part C performance notebook does not overwrite the original Part B metrics file. It creates the separate bootstrap confidence interval output required for the Part C statistical analysis.

---

## 13. Output Files

The Model 2 performance analysis uses and produces the following files.

| File                                                             | Purpose                                                    |
| ---------------------------------------------------------------- | ---------------------------------------------------------- |
| `data/processed/returns_features.csv`                            | Input feature-engineered dataset                           |
| `models/model2_xgboost.joblib`                                   | Saved XGBoost model pipeline                               |
| `experimental_results/model2_xgboost_metrics.csv`                | Model 2 metrics generated during Part B                    |
| `experimental_results/model2_bootstrap_confidence_intervals.csv` | Bootstrap 95% confidence intervals generated during Part C |

The Part B metrics file provides the original Model 2 performance results, while the Part C bootstrap file provides statistical uncertainty estimates for the performance metrics.

---

## 14. Validation Checks

The notebook performs validation checks to confirm that:

* The input dataset exists.
* The target variable is present.
* The target variable does not contain missing values.
* The saved Model 2 file exists.
* The model loads successfully.
* The loaded pipeline contains a preprocessing component.
* The loaded pipeline contains a model component.
* The model has a `predict()` method.
* The model has a `predict_proba()` method.
* The number of predictions matches the number of test observations.
* The number of probability predictions matches the number of test observations.
* The bootstrap output file is successfully saved.

These checks help identify missing files, invalid model files, and prediction-size mismatches before the results are finalised.

---

## 15. Model 2 Performance Workflow

```text
Feature-engineered dataset
            |
            v
Recreate stratified 80/20 test split
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
Generate confusion matrix summary
            |
            v
Perform 1,000 bootstrap iterations
            |
            v
Calculate 95% confidence intervals
            |
            v
Save statistical results
```

The workflow uses the saved Model 2 pipeline rather than retraining the XGBoost model during the Part C performance evaluation.

---

## 16. Summary

Model 2 performance is evaluated using the test portion of the feature-engineered STADIOalot dataset.

The evaluation includes standard classification metrics, a classification report, confusion matrix analysis, and bootstrap 95% confidence intervals.

The main performance metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

The statistical analysis uses:

* 1,000 bootstrap iterations
* Sampling with replacement
* A 95% confidence level
* Bootstrap random state `42`

The main statistical output produced during Part C is:

```text
experimental_results/model2_bootstrap_confidence_intervals.csv
```

The original Model 2 performance metrics from Part B are stored separately in:

```text
experimental_results/model2_xgboost_metrics.csv
```

The associated Model 2 performance notebook is:

```text
notebooks/Model2Performance.ipynb
```

This notebook should be executed after the Model 2 training notebook has successfully created:

```text
models/model2_xgboost.joblib
```
