# Model Comparison — Random Forest vs XGBoost

## Comparison of Model 1 and Model 2

This document describes the comparative evaluation of the two predictive models developed for the **STADIOalot Return Prediction** project:

* **Model 1 — Random Forest Classifier**
* **Model 2 — XGBoost Classifier**

The purpose of the comparison is to evaluate the predictive performance of both models using the same feature-engineered dataset and the same stratified train-test split.

Both models are evaluated on exactly the same test observations. This ensures that the comparison is fair, reproducible and suitable for paired statistical analysis.

The comparison considers:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion matrices
* Bootstrap 95% confidence intervals
* Paired bootstrap confidence intervals for model differences
* McNemar's statistical test
* ROC curve comparison

---

# 1. Models Being Compared

## Model 1 — Random Forest

Model 1 uses a **Random Forest Classifier**.

The trained model is stored at:

```text
models/model1_random_forest.joblib
```

Its performance notebook is:

```text
notebooks/Model1Performance.ipynb
```

Its calculated performance metrics are stored at:

```text
experimental_results/model1_random_forest_metrics.csv
```

---

## Model 2 — XGBoost

Model 2 uses an **XGBoost Classifier**.

The trained model is stored at:

```text
models/model2_xgboost.joblib
```

Its performance notebook is:

```text
notebooks/Model2Performance.ipynb
```

Its calculated performance metrics are stored at:

```text
experimental_results/model2_xgboost_metrics.csv
```

---

# 2. Comparison Notebook

The complete Model 1 versus Model 2 comparison is implemented in:

```text
notebooks/Comparison.ipynb
```

The comparison notebook:

1. Loads the feature-engineered dataset.
2. Creates a common stratified 80/20 train-test split.
3. Loads the saved Random Forest pipeline.
4. Loads the saved XGBoost pipeline.
5. Generates predictions for the same test observations.
6. Calculates performance metrics.
7. Compares confusion matrices.
8. Generates a performance comparison chart.
9. Generates a combined ROC curve.
10. Calculates bootstrap confidence intervals.
11. Performs paired bootstrap comparison.
12. Performs McNemar's test.
13. Saves the comparison results to the `experimental_results` folder.
14. Verifies that all required output files exist.

The comparison notebook loads the trained model pipelines and evaluates them. It does **not** retrain either model.

---

# 3. Dataset Used for Comparison

Both models use the feature-engineered dataset:

```text
data/processed/returns_features.csv
```

The target variable is:

```text
Return_Status
```

The target is encoded as:

```text
0 = Not Returned
1 = Returned
```

The feature-engineered predictor variables are used as the model inputs.

---

# 4. Fair Comparison Methodology

A fair model comparison requires both models to be evaluated using identical observations.

The comparison notebook therefore creates the test set using:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

This produces:

```text
80% = Training Set
20% = Testing Set
```

The following settings are used:

| Setting          | Value           |
| ---------------- | --------------- |
| Train/Test Split | 80% / 20%       |
| Test Size        | 0.20            |
| Random State     | 42              |
| Stratification   | Yes             |
| Target Variable  | `Return_Status` |

Both models are evaluated against the same `X_test` and `y_test` observations.

This is particularly important for the statistical comparison because the predictions from the two models are paired observation-by-observation.

---

# 5. Model Predictions

After loading the two saved model pipelines, predictions are generated using:

```python
y_pred_model1 = model1.predict(X_test)
y_probability_model1 = model1.predict_proba(X_test)[:, 1]

y_pred_model2 = model2.predict(X_test)
y_probability_model2 = model2.predict_proba(X_test)[:, 1]
```

The notebook validates that:

* Both models generated predictions for the same number of test observations.
* Both class-prediction arrays match the size of `y_test`.
* Both probability arrays match the size of `y_test`.

This provides an additional check that both models are being evaluated on the same test set.

---

# 6. Performance Metrics

The following classification metrics are calculated for both models:

| Metric    | Purpose                                                                              |
| --------- | ------------------------------------------------------------------------------------ |
| Accuracy  | Measures the overall proportion of correctly classified observations                 |
| Precision | Measures the proportion of predicted returned orders that were actually returned     |
| Recall    | Measures the proportion of actual returned orders that were correctly identified     |
| F1-Score  | Provides a balance between precision and recall                                      |
| ROC-AUC   | Measures the model's ability to distinguish between returned and non-returned orders |

The metrics are calculated using the test-set predictions.

---

# 7. Accuracy

Accuracy measures the proportion of all test observations that were classified correctly.

It is calculated as:

```text
Accuracy =
(TP + TN) / (TP + TN + FP + FN)
```

Accuracy provides an overall measure of classification correctness.

However, accuracy is not interpreted in isolation because the distribution of returned and non-returned orders should also be considered.

---

# 8. Precision

Precision measures the proportion of observations predicted as returned that were actually returned.

It is calculated as:

```text
Precision =
TP / (TP + FP)
```

A higher precision means that a larger proportion of the orders predicted as returned were actually returned.

For the STADIOalot business problem, precision is relevant because incorrect return predictions could result in unnecessary customer-retention or intervention activities.

---

# 9. Recall

Recall measures the proportion of actual returned orders that were successfully identified by the model.

It is calculated as:

```text
Recall =
TP / (TP + FN)
```

A higher recall means that the model identifies a larger proportion of the orders that ultimately result in returns.

Recall is therefore relevant when the business wants to identify as many potential returns as possible before they occur.

---

# 10. F1-Score

The F1-score combines precision and recall into a single metric.

It is calculated as:

```text
F1-Score =
2 × (Precision × Recall) / (Precision + Recall)
```

F1-score is useful when both false-positive and false-negative classification errors are important.

For this project, F1-score provides a balanced view of the model's ability to identify returned orders while limiting incorrect return predictions.

---

# 11. ROC-AUC

ROC-AUC measures the model's ability to distinguish between the two target classes across different classification thresholds.

The two classes are:

```text
Returned
Not Returned
```

The comparison notebook calculates ROC-AUC using the predicted probability of the positive class:

```python
roc_auc_score(
    y_test,
    y_probability
)
```

A larger ROC-AUC value indicates stronger discrimination between the two classes.

ROC-AUC is considered alongside the other classification metrics rather than being interpreted independently.

---

# 12. Performance Comparison Table

The comparison notebook generates a side-by-side performance table containing:

| Metric    | Model 1 — Random Forest | Model 2 — XGBoost | Difference | Better Model |
| --------- | ----------------------: | ----------------: | ---------: | ------------ |
| Accuracy  |              Calculated |        Calculated |    M2 − M1 | Calculated   |
| Precision |              Calculated |        Calculated |    M2 − M1 | Calculated   |
| Recall    |              Calculated |        Calculated |    M2 − M1 | Calculated   |
| F1-Score  |              Calculated |        Calculated |    M2 − M1 | Calculated   |
| ROC-AUC   |              Calculated |        Calculated |    M2 − M1 | Calculated   |

The numerical values are generated automatically when `Comparison.ipynb` is executed.

The difference is calculated as:

```text
Model 2 score − Model 1 score
```

Therefore:

```text
Positive difference → Model 2 has the higher score
Negative difference → Model 1 has the higher score
Zero difference → Equal score
```

The `Better Model` column identifies the model with the higher value for that particular metric.

This column describes the observed metric value only and does not by itself establish statistical significance.

---

# 13. Confusion Matrix Comparison

Confusion matrices are calculated for both models.

The following values are compared:

| Measure              | Model 1 — Random Forest | Model 2 — XGBoost |
| -------------------- | ----------------------: | ----------------: |
| True Negatives (TN)  |              Calculated |        Calculated |
| False Positives (FP) |              Calculated |        Calculated |
| False Negatives (FN) |              Calculated |        Calculated |
| True Positives (TP)  |              Calculated |        Calculated |

The confusion matrices provide information about the types of classification errors made by each model.

---

# 14. Business Interpretation of the Confusion Matrix

## True Positive

A True Positive occurs when:

```text
Actual = Returned
Prediction = Returned
```

The model correctly identifies an order that ultimately results in a return.

---

## False Positive

A False Positive occurs when:

```text
Actual = Not Returned
Prediction = Returned
```

The model incorrectly predicts that an order will be returned.

In a business context, this could result in unnecessary intervention or retention activity.

---

## False Negative

A False Negative occurs when:

```text
Actual = Returned
Prediction = Not Returned
```

The model fails to identify an order that ultimately results in a return.

This type of error may be important because the business could miss an opportunity to take preventative action.

---

## True Negative

A True Negative occurs when:

```text
Actual = Not Returned
Prediction = Not Returned
```

The model correctly identifies an order that does not result in a return.

---

# 15. Performance Metric Comparison Chart

The notebook produces a bar chart comparing the following metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

The chart provides a visual comparison of the observed metric values for:

```text
Model 1 — Random Forest
Model 2 — XGBoost
```

All metrics are displayed on a common 0-to-1 scale.

---

# 16. ROC Curve Comparison

The notebook generates a combined ROC curve for both models.

The ROC curve compares:

```text
False Positive Rate
```

against:

```text
True Positive Rate
```

The graph includes the ROC-AUC value for each model.

A diagonal reference line representing random classification is also displayed.

The ROC curves allow the discriminatory behaviour of the two models to be visually compared across classification thresholds.

---

# 17. Bootstrap Confidence Intervals

Bootstrap resampling is used to estimate the uncertainty associated with the observed model performance metrics.

The comparison uses:

```text
Number of bootstrap samples = 1,000
Confidence level = 95%
Random seed = 42
```

For each bootstrap sample, the following metrics are recalculated:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

The confidence interval is estimated using the percentile method:

```text
Lower limit = 2.5th percentile
Upper limit = 97.5th percentile
```

The resulting interval provides an estimate of the uncertainty surrounding the observed performance score.

---

# 18. Individual Model Bootstrap Analysis

Bootstrap confidence intervals are calculated separately for:

```text
Model 1 — Random Forest
Model 2 — XGBoost
```

For every model and metric, the output contains:

| Column              | Description                        |
| ------------------- | ---------------------------------- |
| `Analysis`          | Type of bootstrap analysis         |
| `Model`             | Model being evaluated              |
| `Metric`            | Performance metric                 |
| `Observed Score`    | Original test-set metric           |
| `95% CI Lower`      | Lower confidence interval boundary |
| `95% CI Upper`      | Upper confidence interval boundary |
| `Bootstrap Samples` | Number of valid bootstrap samples  |

The results are saved in:

```text
experimental_results/model_comparison_bootstrap_confidence_intervals.csv
```

---

# 19. Paired Bootstrap Comparison

Because both models are evaluated on the same test observations, a paired bootstrap comparison is also performed.

The same bootstrap indices are used for both models for each resampling iteration.

For every bootstrap sample, the difference is calculated as:

```text
Model 2 − Model 1
```

This is performed for:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

A 95% confidence interval is then calculated for each performance difference.

---

# 20. Interpretation of Paired Bootstrap Confidence Intervals

The paired bootstrap results can be interpreted using the location of the confidence interval relative to zero.

| Confidence Interval     | Interpretation                                                              |
| ----------------------- | --------------------------------------------------------------------------- |
| Entire interval above 0 | The observed difference provides evidence favouring Model 2 for that metric |
| Entire interval below 0 | The observed difference provides evidence favouring Model 1 for that metric |
| Interval includes 0     | The difference is uncertain based on the bootstrap interval                 |

The bootstrap analysis provides statistical evidence about the uncertainty of the observed metric differences.

It should be considered together with the actual performance metrics and other statistical evidence.

---

# 21. McNemar's Test

McNemar's test is used to compare the paired classification outcomes of the two models.

The test is appropriate because both models make predictions for exactly the same test observations.

Rather than treating the predictions as independent samples, McNemar's test examines whether the models have different patterns of paired classification errors.

The paired outcomes are divided into four categories:

1. Both models correctly classify the observation.
2. Model 1 is correct and Model 2 is incorrect.
3. Model 1 is incorrect and Model 2 is correct.
4. Both models are incorrect.

---

# 22. McNemar Contingency Table

The notebook creates a 2 × 2 contingency table:

|                       |                     Model 2 Correct |                   Model 2 Incorrect |
| --------------------- | ----------------------------------: | ----------------------------------: |
| **Model 1 Correct**   |                        Both Correct | Model 1 Correct / Model 2 Incorrect |
| **Model 1 Incorrect** | Model 1 Incorrect / Model 2 Correct |                      Both Incorrect |

The two off-diagonal cells represent the observations where the models disagree.

These paired disagreements form the basis of McNemar's test.

---

# 23. McNemar's Test Hypotheses

The significance level is:

```text
α = 0.05
```

The hypotheses are:

```text
H₀:
There is no statistically significant difference
in the paired classification error patterns
of the two models.
```

```text
H₁:
There is a statistically significant difference
in the paired classification error patterns
of the two models.
```

The decision rule is:

```text
p-value < 0.05
    → Reject H₀
    → Evidence of a statistically significant difference

p-value ≥ 0.05
    → Fail to reject H₀
    → Insufficient evidence of a statistically significant difference
```

The actual p-value generated by `Comparison.ipynb` must be reported in the final analysis.

The p-value is not hard-coded in the notebook.

---

# 24. Why McNemar's Test Is Appropriate

McNemar's test is appropriate because the two models are evaluated on paired observations.

The comparison follows:

```text
                 Same Test Set
                      |
          +-----------+-----------+
          |                       |
          v                       v
    Random Forest             XGBoost
     Predictions             Predictions
          |                       |
          +-----------+-----------+
                      |
                      v
             Paired Comparison
```

For example, if both models make predictions for the same 10,000 test orders, the prediction from Model 1 can be paired directly with the prediction from Model 2 for every order.

This paired structure is what makes McNemar's test suitable for examining differences in classification error patterns.

---

# 25. Overall Evaluation Framework

The model comparison considers multiple forms of evidence rather than relying on one metric.

The evaluation includes:

```text
Predictive Performance
        |
        +-- Accuracy
        +-- Precision
        +-- Recall
        +-- F1-Score
        +-- ROC-AUC
        |
        v
Classification Errors
        |
        +-- Confusion Matrix
        |
        v
Uncertainty
        |
        +-- Bootstrap 95% Confidence Intervals
        +-- Paired Bootstrap Differences
        |
        v
Paired Statistical Comparison
        |
        +-- McNemar's Test
```

This framework allows the observed predictive performance and statistical evidence to be considered together.

---

# 26. Interpretation of Model Differences

A higher value for a particular metric indicates that the corresponding model achieved the higher observed score for that metric.

However, a numerical difference does not automatically establish that the difference is statistically significant.

For example:

```text
Model 1 F1-Score = 0.72
Model 2 F1-Score = 0.73
```

The difference is:

```text
0.73 − 0.72 = 0.01
```

This shows that Model 2 has the higher observed F1-score in this example.

However, the size of the difference alone does not determine whether the difference is statistically meaningful.

The bootstrap confidence interval and McNemar's test provide additional statistical evidence.

---

# 27. Model Comparison Results

The primary comparison results are saved to:

```text
experimental_results/model_comparison_metrics.csv
```

The file contains:

| Column                    | Description                          |
| ------------------------- | ------------------------------------ |
| `Metric`                  | Performance metric                   |
| `Model 1 — Random Forest` | Model 1 score                        |
| `Model 2 — XGBoost`       | Model 2 score                        |
| `Difference`              | Model 2 score minus Model 1 score    |
| `Better Model`            | Model with the higher observed score |

---

# 28. Metric Difference Results

The metric difference analysis is saved to:

```text
experimental_results/model_metric_differences.csv
```

The table contains:

| Column                      | Description                                        |
| --------------------------- | -------------------------------------------------- |
| `Metric`                    | Performance metric                                 |
| `Model 1 — Random Forest`   | Model 1 score                                      |
| `Model 2 — XGBoost`         | Model 2 score                                      |
| `Absolute Difference`       | Absolute difference between the scores             |
| `Percentage Difference (%)` | Relative difference using Model 1 as the reference |

The percentage difference is calculated as:

```text
((Model 2 − Model 1) / Model 1) × 100
```

when the Model 1 score is non-zero.

---

# 29. Bootstrap Comparison Output

The combined bootstrap results are saved to:

```text
experimental_results/model_comparison_bootstrap_confidence_intervals.csv
```

The file contains both:

1. Individual model bootstrap confidence intervals.
2. Paired bootstrap confidence intervals for Model 2 − Model 1.

The paired comparison includes:

| Column              | Description                                   |
| ------------------- | --------------------------------------------- |
| `Analysis`          | Identifies the analysis type                  |
| `Model`             | Model or model comparison                     |
| `Metric`            | Performance metric                            |
| `Observed Score`    | Observed metric or observed difference        |
| `95% CI Lower`      | Lower confidence interval boundary            |
| `95% CI Upper`      | Upper confidence interval boundary            |
| `Bootstrap Samples` | Number of bootstrap samples                   |
| `CI Includes Zero`  | Whether the difference interval contains zero |
| `Interpretation`    | Interpretation of the interval                |

---

# 30. McNemar Test Output

The McNemar test results are saved to:

```text
experimental_results/mcnemar_test.csv
```

The output contains:

| Column                                | Description                                                  |
| ------------------------------------- | ------------------------------------------------------------ |
| `Test`                                | Statistical test used                                        |
| `Model 1 Correct / Model 2 Incorrect` | Number of paired disagreements favouring Model 1             |
| `Model 1 Incorrect / Model 2 Correct` | Number of paired disagreements favouring Model 2             |
| `Both Correct`                        | Number of observations correctly classified by both models   |
| `Both Incorrect`                      | Number of observations incorrectly classified by both models |
| `Test Statistic`                      | McNemar test statistic                                       |
| `p-value`                             | Statistical significance value                               |
| `Alpha`                               | Significance threshold                                       |
| `Decision`                            | Decision based on the p-value                                |

---

# 31. Overall Evaluation Output

The notebook also produces an overall evaluation table containing:

| Decision Area       | Result                           |
| ------------------- | -------------------------------- |
| Primary Criterion   | F1-Score                         |
| Primary Result      | Calculated from observed results |
| Secondary Criterion | ROC-AUC                          |
| Secondary Result    | Calculated from observed results |
| Statistical Test    | McNemar's Test                   |
| Significance Level  | α = 0.05                         |
| McNemar Result      | Calculated from p-value          |

The notebook uses F1-score as the primary evaluation criterion because it considers both precision and recall.

ROC-AUC provides an additional assessment of discriminatory performance.

The final interpretation must use the actual results produced by the notebook.

---

# 32. Important Interpretation Rule

Model selection should not be based on a single numerical value without considering the broader evaluation results.

For example:

```text
Higher Accuracy
```

does not automatically mean that a model has better recall.

Similarly:

```text
Higher ROC-AUC
```

does not automatically mean that the model has better precision or recall at the selected classification threshold.

The following should therefore be considered together:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion matrix
* Bootstrap confidence intervals
* Paired bootstrap differences
* McNemar's test

The final project report should describe the observed differences and the statistical evidence rather than relying on a single metric.

---

# 33. Reproducibility

The comparison uses the following reproducibility settings:

```text
Train/Test Split = 80% / 20%
Random State = 42
Stratification = Yes
Bootstrap Samples = 1,000
Bootstrap Random Seed = 42
Confidence Level = 95%
Significance Level = 0.05
```

The same dataset, saved model pipelines, test split, and random seeds should produce reproducible comparison results.

---

# 34. Statistical Software

The comparison notebook uses Python and the following libraries:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
statsmodels
joblib
```

`statsmodels` is used for McNemar's test.

If `statsmodels` is not installed, it can be installed using:

```bash
pip install statsmodels
```

---

# 35. Comparison Output Files

The following Part C comparison files are generated:

```text
experimental_results/
│
├── model_comparison_metrics.csv
├── model_comparison_bootstrap_confidence_intervals.csv
├── model_metric_differences.csv
└── mcnemar_test.csv
```

The individual model performance files are also maintained:

```text
experimental_results/
│
├── model1_random_forest_metrics.csv
├── model1_bootstrap_confidence_intervals.csv
├── model2_xgboost_metrics.csv
└── model2_bootstrap_confidence_intervals.csv
```

---

# 36. Complete Part C File Structure

The relevant Part C project structure is:

```text
STADIOalot-Return-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│       └── returns_features.csv
│
├── models/
│   ├── model1_random_forest.joblib
│   └── model2_xgboost.joblib
│
├── notebooks/
│   ├── Model1Performance.ipynb
│   ├── Model2Performance.ipynb
│   └── Comparison.ipynb
│
├── experimental_results/
│   ├── model1_random_forest_metrics.csv
│   ├── model1_bootstrap_confidence_intervals.csv
│   ├── model2_xgboost_metrics.csv
│   ├── model2_bootstrap_confidence_intervals.csv
│   ├── model_comparison_metrics.csv
│   ├── model_comparison_bootstrap_confidence_intervals.csv
│   ├── model_metric_differences.csv
│   └── mcnemar_test.csv
│
└── docs/
    └── Comparison.MD
```

---

# 37. File Relationships

The complete Part C comparison workflow is:

```text
                     FEATURE-ENGINEERED DATA
                              |
                              v
             data/processed/returns_features.csv
                              |
                 +------------+------------+
                 |                         |
                 v                         v
       Model 1 Random Forest       Model 2 XGBoost
                 |                         |
                 v                         v
 model1_random_forest.joblib     model2_xgboost.joblib
                 |                         |
                 v                         v
      Model1Performance.ipynb     Model2Performance.ipynb
                 |                         |
                 v                         v
 model1_random_forest_metrics.csv model2_xgboost_metrics.csv
                 |                         |
                 +------------+------------+
                              |
                              v
                    Comparison.ipynb
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
 Performance Metrics   Statistical Analysis   Visual Analysis
          |                   |                   |
          |                   +-- Bootstrap       +-- Performance Chart
          |                   +-- Paired Bootstrap+-- ROC Curves
          |                   +-- McNemar         +-- Confusion Matrices
          |                   | 
          +-------------------+-------------------+
                              |
                              v
                  experimental_results/
                              |
       +----------------------+----------------------+
       |                      |                      |
       v                      v                      v
model_comparison_      model_metric_          mcnemar_test.csv
metrics.csv            differences.csv
                              |
                              v
               model_comparison_bootstrap_
               confidence_intervals.csv
```

---

# 38. How to Run the Complete Part C Workflow

The recommended execution order is:

## Step 1 — Model 1 Performance

Open:

```text
notebooks/Model1Performance.ipynb
```

Run all cells.

The notebook should generate:

```text
experimental_results/model1_random_forest_metrics.csv
```

and the associated Model 1 bootstrap results.

---

## Step 2 — Model 2 Performance

Open:

```text
notebooks/Model2Performance.ipynb
```

Run all cells.

The notebook should generate:

```text
experimental_results/model2_xgboost_metrics.csv
```

and the associated Model 2 bootstrap results.

---

## Step 3 — Model Comparison

Open:

```text
notebooks/Comparison.ipynb
```

Run all cells.

The notebook generates:

```text
experimental_results/model_comparison_metrics.csv
```

```text
experimental_results/model_comparison_bootstrap_confidence_intervals.csv
```

```text
experimental_results/model_metric_differences.csv
```

```text
experimental_results/mcnemar_test.csv
```

The notebook also generates the comparison tables, confusion-matrix results, performance chart, ROC curve, bootstrap analysis, and McNemar test.

---

# 39. Part C Requirement Coverage

The comparison implementation addresses the Part C requirements as follows:

| Part C Requirement                | Evidence                                              |
| --------------------------------- | ----------------------------------------------------- |
| Performance of Model 1            | `Model1Performance.ipynb`                             |
| Performance of Model 2            | `Model2Performance.ipynb`                             |
| Comparison of Model 1 and Model 2 | `Comparison.ipynb`                                    |
| Accuracy                          | Calculated for both models                            |
| Precision                         | Calculated for both models                            |
| Recall                            | Calculated for both models                            |
| F1-Score                          | Calculated for both models                            |
| ROC-AUC                           | Calculated for both models                            |
| Confusion Matrix                  | Calculated for both models                            |
| Statistical measures              | Bootstrap 95% confidence intervals                    |
| Paired statistical comparison     | Paired bootstrap comparison                           |
| Statistical test of paired errors | McNemar's test                                        |
| Fair model comparison             | Same stratified test observations                     |
| Reproducibility                   | `random_state=42`                                     |
| Saved Model 1 results             | `model1_random_forest_metrics.csv`                    |
| Saved Model 2 results             | `model2_xgboost_metrics.csv`                          |
| Saved comparison results          | `model_comparison_metrics.csv`                        |
| Saved bootstrap results           | `model_comparison_bootstrap_confidence_intervals.csv` |
| Saved metric differences          | `model_metric_differences.csv`                        |
| Saved McNemar results             | `mcnemar_test.csv`                                    |

---

# 40. Final Evaluation Principle

The final comparison should consider both predictive performance and statistical evidence.

F1-score provides a useful primary measure because it balances precision and recall.

ROC-AUC provides an additional assessment of the models' ability to discriminate between returned and non-returned orders.

Accuracy, precision, recall, and the confusion matrices provide further information about classification performance and error types.

Bootstrap confidence intervals provide information about uncertainty around the observed performance measures and their paired differences.

McNemar's test evaluates whether the paired classification error patterns differ statistically at:

```text
α = 0.05
```

The final interpretation should therefore be based on the **actual numerical results generated by `Comparison.ipynb`**.

No model should be selected in advance. The reported interpretation should reflect the observed performance metrics, confidence intervals, confusion matrices, and statistical test results produced when the notebook is executed.
