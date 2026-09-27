# Model 1 Performance — Random Forest Classifier

## 1. Overview

This document presents the performance analysis of **Model 1**, which uses a **Random Forest Classifier** to predict whether an e-commerce order will be returned.

The model was developed for the **STADIOalot Return Prediction** project. The performance notebook evaluates the trained Random Forest model on a recreated stratified test set generated from the feature-engineered dataset.

The performance analysis includes:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Classification report
* Confusion matrix
* Bootstrap 95% confidence intervals

The bootstrap analysis uses **1,000 iterations** to estimate 95% confidence intervals for the observed performance metrics.

---

## 2. Model 1 Details

| Item                 | Description                           |
| -------------------- | ------------------------------------- |
| Model                | Model 1                               |
| Algorithm            | Random Forest Classifier              |
| Problem Type         | Binary classification                 |
| Target Variable      | `Return_Status`                       |
| Class `0`            | Not Returned                          |
| Class `1`            | Returned                              |
| Dataset              | Feature-engineered STADIOalot dataset |
| Train/Test Split     | 80% training and 20% testing          |
| Random State         | `42`                                  |
| Performance Notebook | `05_Model1Performance.ipynb`          |

The performance notebook loads the feature-engineered dataset, recreates the stratified 80/20 train/test split, loads the saved Random Forest pipeline, generates predictions, calculates performance metrics, and performs bootstrap analysis.

---

## 3. Input Dataset

The performance analysis uses the feature-engineered dataset generated during the data preparation and feature engineering stages.

```text
data/processed/returns_features.csv
```

The target variable is:

```text
Return_Status
```

The target variable is encoded as follows:

| Value | Meaning      |
| ----: | ------------ |
|   `0` | Not Returned |
|   `1` | Returned     |

The target column is separated from the predictor variables before the train/test split is recreated.

The notebook also verifies that:

* The input dataset exists.
* `Return_Status` is present.
* `Return_Status` does not contain missing values.

---

## 4. Trained Model

The saved Random Forest pipeline is loaded from:

```text
models/model1_random_forest.joblib
```

The saved pipeline contains the trained preprocessing and modelling components required to generate predictions.

The notebook verifies that the loaded model provides:

* `predict()`
* `predict_proba()`

These methods are used to generate class predictions and probabilities for the `Returned` class.

---

## 5. Performance Analysis Notebook

The Model 1 performance analysis is implemented in:

```text
05_Model1Performance.ipynb
```

The notebook performs the following operations:

1. Defines the project paths.
2. Loads the feature-engineered dataset.
3. Validates the target variable.
4. Separates predictor variables and target values.
5. Displays the target class distribution.
6. Recreates the stratified 80/20 train/test split.
7. Verifies the target distribution after splitting.
8. Loads the saved Random Forest pipeline.
9. Validates the loaded model.
10. Generates class predictions.
11. Generates return probabilities.
12. Calculates performance metrics.
13. Generates a classification report.
14. Generates a confusion matrix summary.
15. Performs 1,000 bootstrap iterations.
16. Calculates 95% bootstrap confidence intervals.
17. Saves the bootstrap confidence interval results.
18. Displays a final Model 1 statistical output summary.

---

## 6. Project Paths

The notebook dynamically identifies the project root by searching upwards until a directory containing the `data` folder is found.

The main paths used by the notebook are:

| File / Folder                                                    | Purpose                              |
| ---------------------------------------------------------------- | ------------------------------------ |
| `data/processed/returns_features.csv`                            | Feature-engineered input dataset     |
| `models/model1_random_forest.joblib`                             | Saved Model 1 Random Forest pipeline |
| `experimental_results/`                                          | Directory for experimental results   |
| `experimental_results/model1_random_forest_metrics.csv`          | Existing Model 1 performance metrics |
| `experimental_results/model1_bootstrap_confidence_intervals.csv` | Bootstrap confidence interval output |

The notebook creates the `experimental_results` directory if it does not already exist.

---

## 7. How to Run the Notebook

Run the notebook from the project environment.

### Step 1: Install the Required Libraries

If the project uses the requirements file stored in the experimental setup folder, install the dependencies using:

```bash
pip install -r experimental_setup/requirements.txt
```

### Step 2: Open the Notebook

The notebook can be opened using Jupyter Notebook or the Jupyter extension in Visual Studio Code.

```bash
jupyter notebook
```

Open:

```text
05_Model1Performance.ipynb
```

### Step 3: Execute the Notebook

Run the notebook cells from top to bottom.

The following files must exist before the notebook is executed:

```text
data/processed/returns_features.csv
models/model1_random_forest.joblib
```

The feature-engineered dataset is produced by the preceding feature-engineering stage, while the trained Random Forest model is produced by the Model 1 training notebook.

---

## 8. Target Class Distribution

The notebook calculates the frequency and percentage of each target class before recreating the train/test split.

The classes are:

| Return Status | Class        |
| ------------: | ------------ |
|           `0` | Not Returned |
|           `1` | Returned     |

The target distribution table contains:

* `Return_Status`
* `Class`
* `Frequency`
* `Percentage`

This provides a summary of the class distribution in the complete feature-engineered dataset.

---

## 9. Recreating the Model 1 Test Set

The notebook recreates the Model 1 train/test split using a stratified 80/20 split.

The split configuration is:

| Parameter           | Value |
| ------------------- | ----: |
| Training proportion |   80% |
| Testing proportion  |   20% |
| Random state        |  `42` |
| Stratification      |   Yes |

The split is implemented using:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Stratification is used so that the class distribution of `Return_Status` is maintained as closely as possible across the training and testing sets.

The notebook displays:

* Full dataset size
* Training set size
* Testing set size
* Training class counts
* Training class percentages
* Testing class counts
* Testing class percentages

The recreated test set is then used for model evaluation.

---

## 10. Loading and Validating the Saved Model

The trained Random Forest pipeline is loaded using `joblib`.

```python
model1 = joblib.load(MODEL_PATH)
```

Before predictions are generated, the notebook verifies that:

* The model file exists.
* The model loads successfully.
* The model provides a `predict()` method.
* The model provides a `predict_proba()` method.

If the model file does not exist, the notebook raises an error instructing the user to run the Model 1 training notebook first.

---

## 11. Prediction Outputs

The loaded Random Forest model produces two prediction outputs.

### 11.1 Class Predictions

Class predictions are generated using:

```python
y_pred = model1.predict(X_test)
```

The predicted values are:

| Prediction | Meaning                |
| ---------: | ---------------------- |
|        `0` | Predicted Not Returned |
|        `1` | Predicted Returned     |

### 11.2 Return Probabilities

The probability of an order belonging to the `Returned` class is generated using:

```python
y_probability = model1.predict_proba(X_test)[:, 1]
```

This probability represents the model's estimated probability for the `Returned` class.

The notebook validates that both the class predictions and probability predictions contain the same number of observations as the test set.

---

## 12. Prediction Preview

The notebook creates a prediction preview containing:

| Column                    | Description                                   |
| ------------------------- | --------------------------------------------- |
| `Actual Return Status`    | Actual target value                           |
| `Actual Class`            | Text representation of the actual class       |
| `Predicted Return Status` | Model prediction                              |
| `Predicted Class`         | Text representation of the predicted class    |
| `Return Probability`      | Predicted probability of the `Returned` class |

The first 10 predictions are displayed as a preview.

---

## 13. Performance Metrics

The notebook calculates five primary performance metrics.

### 13.1 Accuracy

Accuracy measures the overall proportion of correctly classified observations.

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

The notebook calculates accuracy using:

```python
accuracy_score(y_test, y_pred)
```

---

### 13.2 Precision

Precision measures the proportion of observations predicted as returned that were actually returned.

$$
Precision = \frac{TP}{TP + FP}
$$

The notebook calculates precision using:

```python
precision_score(
    y_test,
    y_pred,
    zero_division=0
)
```

---

### 13.3 Recall

Recall measures the proportion of actual returned orders that were correctly identified.

$$
Recall = \frac{TP}{TP + FN}
$$

The notebook calculates recall using:

```python
recall_score(
    y_test,
    y_pred,
    zero_division=0
)
```

---

### 13.4 F1-Score

The F1-score is the harmonic mean of precision and recall.

$$
F1 = 2 \times
\frac{Precision \times Recall}
{Precision + Recall}
$$

The notebook calculates F1-score using:

```python
f1_score(
    y_test,
    y_pred,
    zero_division=0
)
```

---

### 13.5 ROC-AUC

ROC-AUC measures the model's ability to distinguish between returned and non-returned orders across classification thresholds.

The ROC-AUC calculation uses the predicted probabilities for the `Returned` class:

```python
roc_auc_score(
    y_test,
    y_probability
)
```

Therefore, ROC-AUC is calculated using probability predictions rather than the binary class predictions.

---

## 14. Structured Performance Metrics Table

The notebook creates a structured metrics table containing:

| Metric    | Description                                                      |
| --------- | ---------------------------------------------------------------- |
| Accuracy  | Overall proportion of correct predictions                        |
| Precision | Proportion of predicted returns that were actually returned      |
| Recall    | Proportion of actual returns correctly identified                |
| F1-Score  | Harmonic balance between precision and recall                    |
| ROC-AUC   | Ability to discriminate between returned and non-returned orders |

These values are displayed during notebook execution.

---

## 15. Classification Report

The notebook generates a classification report using:

```python
classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)
```

The resulting report is converted into a DataFrame for display.

The classification table includes results for:

* Not Returned
* Returned
* Macro Average
* Weighted Average
* Accuracy

The class-level results include:

* Precision
* Recall
* F1-score
* Support

The classification report is **displayed in the notebook but is not saved as a separate CSV file by this notebook**.

---

## 16. Confusion Matrix

The notebook calculates the confusion matrix using:

```python
cm = confusion_matrix(
    y_test,
    y_pred
)
```

The four confusion matrix components are extracted:

```text
True Negatives (TN)
False Positives (FP)
False Negatives (FN)
True Positives (TP)
```

They are interpreted as follows:

| Measure             | Interpretation                                        |
| ------------------- | ----------------------------------------------------- |
| True Negative (TN)  | Correctly predicted order that was not returned       |
| False Positive (FP) | Order predicted as returned but actually not returned |
| False Negative (FN) | Order predicted as not returned but actually returned |
| True Positive (TP)  | Correctly predicted returned order                    |

The notebook displays these values in a confusion matrix summary table.

---

## 17. Bootstrap 95% Confidence Intervals

Bootstrap resampling is used to estimate uncertainty around the observed Model 1 performance metrics.

The bootstrap configuration is:

| Parameter                      |                     Value |
| ------------------------------ | ------------------------: |
| Bootstrap iterations requested |                   `1,000` |
| Confidence level               |                     `95%` |
| Bootstrap random state         |                      `42` |
| Resampling method              | Sampling with replacement |
| Resampling unit                |     Test-set observations |

The bootstrap random number generator is created using:

```python
rng = np.random.default_rng(42)
```

For each of the 1,000 iterations, the notebook samples test-set indices with replacement.

The following metrics are recalculated for each bootstrap sample:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

### ROC-AUC Handling

ROC-AUC requires both target classes to be present in the bootstrap sample.

The notebook therefore checks:

```python
if len(np.unique(y_true_bootstrap)) == 2:
```

Only bootstrap samples containing both classes contribute to the ROC-AUC bootstrap distribution.

This means that the number of valid ROC-AUC bootstrap samples can be lower than the requested 1,000 iterations if one-class bootstrap samples occur.

### Confidence Interval Calculation

The 95% confidence interval is calculated using the:

* 2.5th percentile as the lower boundary
* 97.5th percentile as the upper boundary

The notebook records the number of valid bootstrap samples for each metric.

---

## 18. Bootstrap Statistical Output

The bootstrap results are saved to:

```text
experimental_results/model1_bootstrap_confidence_intervals.csv
```

The output contains:

| Column                           | Description                              |
| -------------------------------- | ---------------------------------------- |
| `Model`                          | Model identifier                         |
| `Algorithm`                      | Algorithm used                           |
| `Metric`                         | Performance metric                       |
| `Observed Score`                 | Original test-set metric                 |
| `95% CI Lower`                   | Lower confidence interval boundary       |
| `95% CI Upper`                   | Upper confidence interval boundary       |
| `Bootstrap Samples`              | Number of valid bootstrap samples        |
| `Bootstrap Iterations Requested` | Requested number of bootstrap iterations |
| `Random State`                   | Bootstrap random state                   |

For this notebook, the algorithm is recorded as:

```text
Random Forest
```

and the model is recorded as:

```text
Model 1
```

---

## 19. Model 1 Metrics File

The original Model 1 performance metrics file is:

```text
experimental_results/model1_random_forest_metrics.csv
```

This file is **not created or overwritten by the performance notebook's bootstrap section**.

The notebook uses the existing Model 1 metrics as the previously generated performance output, while the current notebook creates the separate bootstrap confidence interval file:

```text
experimental_results/model1_bootstrap_confidence_intervals.csv
```

Therefore, the two statistical outputs are kept separate.

---

## 20. Output Files

The Model 1 performance analysis uses and produces the following files:

| File                                                             | Purpose                                   |
| ---------------------------------------------------------------- | ----------------------------------------- |
| `data/processed/returns_features.csv`                            | Input feature-engineered dataset          |
| `models/model1_random_forest.joblib`                             | Saved Random Forest model pipeline        |
| `experimental_results/model1_random_forest_metrics.csv`          | Existing Model 1 performance metrics      |
| `experimental_results/model1_bootstrap_confidence_intervals.csv` | Bootstrap 95% confidence interval results |

The main output generated by this performance notebook is:

```text
experimental_results/model1_bootstrap_confidence_intervals.csv
```

---

## 21. Validation Checks

The notebook performs validation checks throughout the performance analysis.

These include:

* Checking that the input dataset exists.
* Checking that `Return_Status` exists.
* Checking that `Return_Status` contains no missing values.
* Checking that the saved Model 1 file exists.
* Checking that the saved model loads successfully.
* Checking that the model has a `predict()` method.
* Checking that the model has a `predict_proba()` method.
* Checking that the number of class predictions matches the number of test observations.
* Checking that the number of probability predictions matches the number of test observations.
* Checking that the bootstrap output file is successfully saved.

These checks help identify missing files, invalid model files and prediction-size mismatches before the statistical results are finalised.

---

## 22. Final Model 1 Statistical Summary

At the end of the notebook, a final statistical summary is displayed.

The summary contains:

| Category                   |
| -------------------------- |
| Model                      |
| Algorithm                  |
| Dataset                    |
| Target                     |
| Train/Test Split           |
| Random State               |
| Training Rows              |
| Testing Rows               |
| Accuracy                   |
| Precision                  |
| Recall                     |
| F1-Score                   |
| ROC-AUC                    |
| True Negatives             |
| False Positives            |
| False Negatives            |
| True Positives             |
| Bootstrap Iterations       |
| Bootstrap Confidence Level |
| Bootstrap Output           |

The bootstrap configuration reported in the summary is:

```text
Bootstrap Iterations: 1,000
Bootstrap Confidence Level: 95%
```

The bootstrap output path is:

```text
experimental_results/model1_bootstrap_confidence_intervals.csv
```

---

## 23. Model 1 Performance Workflow

```text
Feature-engineered dataset
            |
            v
Validate dataset and target
            |
            v
Recreate stratified 80/20 train/test split
            |
            v
Verify target distribution
            |
            v
Load saved Random Forest pipeline
            |
            v
Validate loaded model
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
Save bootstrap statistical results
            |
            v
Display final Model 1 statistical summary
```

---

## 24. Execution Order in the Project

The Model 1 performance notebook should be executed after the required Model 1 training outputs have been created.

The relevant workflow is:

```text
01_Preprocessing.ipynb
        |
        v
02_FeatureEngineering.ipynb
        |
        v
Model 1 Training Notebook - (03_Model1_RandomForest.ipynb)
        |
        +----> models/model1_random_forest.joblib
        |
        +----> experimental_results/model1_random_forest_metrics.csv
        |
        v
05_Model1Performance.ipynb
        |
        v
experimental_results/model1_bootstrap_confidence_intervals.csv
```

The performance notebook therefore depends on the feature-engineered dataset and the saved Model 1 pipeline.

---

## 25. Summary

Model 1 performance is evaluated using the test portion of the feature-engineered STADIOalot dataset.

The notebook:

* Loads the feature-engineered dataset.
* Recreates a stratified 80/20 train/test split using `random_state = 42`.
* Loads the saved Random Forest pipeline.
* Generates class predictions and return probabilities.
* Calculates Accuracy, Precision, Recall, F1-Score and ROC-AUC.
* Generates a classification report.
* Generates a confusion matrix summary.
* Performs 1,000 bootstrap iterations.
* Calculates 95% bootstrap confidence intervals.
* Saves the bootstrap confidence interval results.

The performance notebook is:

```text
05_Model1Performance.ipynb
```

The trained Model 1 pipeline is:

```text
models/model1_random_forest.joblib
```

The existing Model 1 metrics file is:

```text
experimental_results/model1_random_forest_metrics.csv
```

The main statistical output generated by this notebook is:

```text
experimental_results/model1_bootstrap_confidence_intervals.csv
```
