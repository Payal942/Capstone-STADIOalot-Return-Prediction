# Model 2 — XGBoost Classifier

## 1. Model 2 Overview

Model 2 is an **XGBoost Classifier** developed to predict whether an STADIOalot e-commerce order will be returned.

The model uses the feature-engineered dataset produced by **Notebook 2 — Feature Engineering**.

The modelling process includes:

* Loading the feature-engineered dataset
* Separating features and target
* Creating an 80/20 stratified train/test split
* Identifying numerical and categorical features
* Applying preprocessing
* Training the XGBoost classifier
* Generating class predictions and return probabilities
* Evaluating model performance
* Generating a classification report
* Generating a confusion matrix
* Saving the trained model
* Saving the evaluation metrics

The target variable is:

```text
Return_Status
```

where:

```text
0 = Not Returned
1 = Returned
```

---

## 2. Model 2 Training and Inference File

The notebook responsible for **Model 2 training and inference** is:

```text
notebooks/04_Model2_XGBoost.ipynb
```

This notebook contains the complete implementation for Model 2.

### Training

The notebook trains the XGBoost model using the training portion of the feature-engineered dataset:

```text
Feature-Engineered Dataset
        ↓
returns_features.csv
        ↓
80/20 Stratified Split
        ↓
Training Data
        ↓
Preprocessing
        ↓
XGBoost Classifier
        ↓
Trained Model Pipeline
```

### Inference

After training, the notebook performs inference on the unseen testing dataset.

It generates:

```text
Class Predictions
Return Probabilities
```

The class prediction identifies whether an order is predicted to be returned, while the probability represents the estimated probability of the `Returned` class.

---

## 3. Model Input

Model 2 uses the feature-engineered dataset:

```text
data/processed/returns_features.csv
```

This dataset is produced by:

```text
notebooks/02_FeatureEngineering.ipynb
```

The target variable is:

```text
Return_Status
```

The feature matrix is created by removing the target column:

```python
X = df.drop(columns=["Return_Status"])
```

The target vector is converted to integer values:

```python
y = df["Return_Status"].astype(int)
```

All remaining columns in `returns_features.csv` are used as predictor variables.

The model uses information available from the feature-engineered dataset and does not use post-return information that was removed during the preprocessing and feature-engineering stages.

---

## 4. Why XGBoost?

XGBoost is a gradient-boosting ensemble learning method based on decision trees.

The trees are constructed sequentially, with subsequent trees helping to improve the predictions of the previous trees.

XGBoost was selected as Model 2 because it is suitable for structured tabular data and can model complex, non-linear relationships between customer, product, transaction and engineered order characteristics.

Using XGBoost as Model 2 also provides a second tree-based approach that can be evaluated using the same feature-engineered dataset and train/test methodology as Model 1.

---

## 5. Train/Test Split

The dataset is divided into:

```text
80% Training Data
20% Testing Data
```

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

A stratified split is used so that the proportion of returned and non-returned orders is approximately maintained in both the training and testing datasets.

The fixed `random_state` of `42` makes the split reproducible.

The same 80/20 stratified approach is used for Random Forest and XGBoost to support a consistent model comparison.

---

## 6. Feature Type Identification

The notebook identifies numerical and categorical features after the train/test split.

Numerical features are identified using:

```python
X_train.select_dtypes(include=np.number)
```

Categorical features are identified using:

```python
X_train.select_dtypes(exclude=np.number)
```

Separate feature inventories are displayed in the notebook.

This allows different preprocessing methods to be applied according to feature type.

---

## 7. Data Preprocessing

The XGBoost model uses a scikit-learn `ColumnTransformer` containing separate numerical and categorical preprocessing pipelines.

The preprocessing is included inside the complete modelling pipeline.

This ensures that preprocessing is fitted using the training data and then applied consistently during inference.

### 7.1 Numerical Features

Numerical features are processed using:

```text
Numerical Features
        ↓
Median Imputation
        ↓
StandardScaler
```

The numerical pipeline uses:

```python
SimpleImputer(strategy="median")
StandardScaler()
```

Median imputation replaces missing numerical values using the median calculated from the training data.

`StandardScaler` standardises the numerical variables.

### 7.2 Categorical Features

Categorical features are processed using:

```text
Categorical Features
        ↓
Most-Frequent Imputation
        ↓
One-Hot Encoding
```

The categorical pipeline uses:

```python
SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore")
```

`handle_unknown="ignore"` ensures that unseen categories encountered during inference do not cause prediction errors.

---

## 8. Complete Preprocessing and Model Pipeline

The preprocessing and XGBoost classifier are combined into a single scikit-learn pipeline.

```text
Input Features
        ↓
ColumnTransformer
        │
        ├───────────────────┐
        ▼                   ▼
    Numerical           Categorical
     Pipeline             Pipeline
        │                   │
        ▼                   ▼
Median Imputation   Most-Frequent Imputation
        │                   │
        ▼                   ▼
StandardScaler      One-Hot Encoding
        │                   │
        └─────────┬─────────┘
                  ▼
          XGBoost Classifier
                  │
                  ▼
          Trained Model Pipeline
```

The complete pipeline is created using:

```python
model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            xgboost
        )
    ]
)
```

Keeping preprocessing and modelling inside one pipeline ensures that the same transformations are applied consistently during training and inference.

---

## 9. XGBoost Hyperparameters

The XGBoost classifier uses the following configuration:

| Parameter          |             Value |
| ------------------ | ----------------: |
| `n_estimators`     |               300 |
| `max_depth`        |                 6 |
| `learning_rate`    |              0.05 |
| `subsample`        |              0.80 |
| `colsample_bytree` |              0.80 |
| `min_child_weight` |                 2 |
| `reg_lambda`       |               1.0 |
| `objective`        | `binary:logistic` |
| `eval_metric`      |         `logloss` |
| `random_state`     |                42 |
| `n_jobs`           |                -1 |

The implementation is:

```python
xgboost = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.80,
    colsample_bytree=0.80,
    min_child_weight=2,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
```

### Number of Estimators

`n_estimators=300` specifies 300 boosting trees.

### Maximum Tree Depth

`max_depth=6` limits the maximum depth of each tree.

### Learning Rate

`learning_rate=0.05` controls the contribution of each boosting iteration.

### Row Subsampling

`subsample=0.80` uses 80% of the training observations for each boosting iteration.

### Feature Subsampling

`colsample_bytree=0.80` uses 80% of the available features for each tree.

### Minimum Child Weight

`min_child_weight=2` controls the minimum amount of observation weight required for a child node.

### L2 Regularisation

`reg_lambda=1.0` applies L2 regularisation to the model.

### Objective Function

`objective="binary:logistic"` configures XGBoost for binary classification and probability prediction.

### Evaluation Metric

`eval_metric="logloss"` specifies log-loss as the XGBoost evaluation metric during model training.

### Random State

`random_state=42` supports reproducibility.

### Number of Jobs

`n_jobs=-1` allows parallel processing using the available CPU cores.

---

## 10. Model Training

The complete pipeline is trained using:

```python
model_pipeline.fit(
    X_train,
    y_train
)
```

The training workflow is:

```text
Training Features
        ↓
Preprocessing
        ↓
XGBoost Training
        ↓
Trained Model Pipeline
```

The testing data remains separate until inference and evaluation.

The notebook displays a training summary confirming that the model has completed training.

---

## 11. Inference

After training, the model generates predictions for the unseen testing dataset:

```text
X_test
```

Two prediction outputs are generated.

### 11.1 Class Predictions

Class predictions are generated using:

```python
y_pred = model_pipeline.predict(X_test)
```

The output classes are:

```text
0 = Not Returned
1 = Returned
```

### 11.2 Return Probabilities

Return probabilities are generated using:

```python
y_probability = (
    model_pipeline
    .predict_proba(X_test)[:, 1]
)
```

The probability represents the estimated probability of the positive class:

```text
Returned
```

The probability output is used for ROC-AUC evaluation.

---

## 12. Prediction Preview

The notebook creates a prediction preview containing:

```text
Actual Return Status
Actual Label
Predicted Return Status
Predicted Label
Return Probability
```

The first ten predictions are displayed.

This allows the actual test-set outcome, predicted outcome and estimated return probability to be inspected together.

---

## 13. Model Evaluation

The XGBoost model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* Classification Report

### Accuracy

Accuracy measures the proportion of test observations classified correctly.

### Precision

Precision measures the proportion of observations predicted as `Returned` that were actually returned.

### Recall

Recall measures the proportion of orders that actually returned that were correctly identified by the model.

### F1-Score

F1-score combines precision and recall into a single measure.

### ROC-AUC

ROC-AUC measures the model's ability to distinguish between returned and non-returned orders across different classification thresholds.

The ROC-AUC calculation uses the predicted return probabilities.

---

## 14. Classification Report

The notebook generates a classification report containing:

```text
Precision
Recall
F1-Score
Support
```

The classes are displayed as:

```text
Not Returned
Returned
```

Aggregate results include:

```text
Accuracy
Macro Average
Weighted Average
```

The classification report is converted into a pandas DataFrame for inspection.

---

## 15. Confusion Matrix

The notebook calculates a confusion matrix containing:

```text
True Negatives
False Positives
False Negatives
True Positives
```

These outcomes represent:

| Outcome        | Meaning                                       |
| -------------- | --------------------------------------------- |
| True Negative  | Correctly predicted Not Returned              |
| False Positive | Predicted Returned when actually Not Returned |
| False Negative | Predicted Not Returned when actually Returned |
| True Positive  | Correctly predicted Returned                  |

A confusion-matrix heatmap is also generated.

The saved visualisation is:

```text
visualisations/model2_xgboost_confusion_matrix.png
```

The notebook also displays a summary table containing the four confusion-matrix outcomes.

---

## 16. Evaluation Metrics Output

The following evaluation metrics are stored:

```text
accuracy
precision
recall
f1_score
roc_auc
```

They are saved to:

```text
experimental_results/model2_xgboost_metrics.csv
```

The saved CSV contains:

```text
model
accuracy
precision
recall
f1_score
roc_auc
```

The model name stored in the metrics output is:

```text
XGBoost
```

---

## 17. Trained Model Output

The complete trained preprocessing and XGBoost pipeline is saved using `joblib`.

The output file is:

```text
models/model2_xgboost.joblib
```

The saved object contains:

```text
Preprocessing
     +
XGBoost Classifier
```

This allows the complete trained pipeline to be reused during future inference.

---

## 18. Visualisation Output

The Model 2 notebook generates and saves a confusion matrix visualisation.

The output file is:

```text
visualisations/model2_xgboost_confusion_matrix.png
```

The image contains labelled axes for:

```text
Not Returned
Returned
```

and displays the classification counts in the matrix.

---

## 19. Notebook Associated With Model 2

The notebook responsible for Model 2 **training and inference** is:

```text
notebooks/04_Model2_XGBoost.ipynb
```

This is the primary executable file for Model 2.

No separate Python training script is required for the documented implementation because the complete training and inference workflow is contained within this notebook.

---

## 20. Notebook Execution Order

Model 2 depends on the outputs of the preceding preprocessing and feature-engineering notebooks.

The recommended execution order is:

```text
01_Preprocessing.ipynb
        ↓
02_FeatureEngineering.ipynb
        ↓
04_Model2_XGBoost.ipynb
```

### Notebook 1

```text
notebooks/01_Preprocessing.ipynb
```

Prepares the cleaned dataset.

### Notebook 2

```text
notebooks/02_FeatureEngineering.ipynb
```

Creates the feature-engineered dataset:

```text
data/processed/returns_features.csv
```

### Notebook 4

```text
notebooks/04_Model2_XGBoost.ipynb
```

Loads the feature-engineered dataset and trains, evaluates and saves Model 2.

---

## 21. How to Run Model 2

Model 2 should be executed from the repository root.

### Step 1 — Install Dependencies

Install the project dependencies:

```bash
pip install -r requirements.txt
```

### Step 2 — Start Jupyter

Start Jupyter Notebook:

```bash
jupyter notebook
```

### Step 3 — Prepare the Input Dataset

Run the previous notebooks in order:

```text
notebooks/01_Preprocessing.ipynb
```

then:

```text
notebooks/02_FeatureEngineering.ipynb
```

Notebook 2 must successfully create:

```text
data/processed/returns_features.csv
```

### Step 4 — Open Model 2

Open:

```text
notebooks/04_Model2_XGBoost.ipynb
```

### Step 5 — Execute the Notebook

Run the cells from top to bottom.

The notebook will:

1. Define the project paths.
2. Load `returns_features.csv`.
3. Validate the `Return_Status` target.
4. Separate features and target.
5. Create the 80/20 stratified train/test split.
6. Verify the target distribution.
7. Identify numerical and categorical features.
8. Configure numerical preprocessing.
9. Configure categorical preprocessing.
10. Combine the preprocessing pipelines.
11. Configure the XGBoost classifier.
12. Build the complete modelling pipeline.
13. Train the XGBoost model.
14. Generate test-set class predictions.
15. Generate return probabilities.
16. Display a prediction preview.
17. Calculate evaluation metrics.
18. Generate the classification report.
19. Generate the confusion matrix.
20. Save the evaluation metrics.
21. Save the trained model pipeline.
22. Perform final validation checks.
23. Display the Model 2 completion summary.

---

## 22. Model 2 Outputs

After successful execution, the following files are produced:

### Trained Model

```text
models/model2_xgboost.joblib
```

### Evaluation Metrics

```text
experimental_results/model2_xgboost_metrics.csv
```

### Confusion Matrix

```text
visualisations/model2_xgboost_confusion_matrix.png
```

The notebook also displays:

```text
Path Summary
Feature Target Summary
Split Summary
Split Target Report
Feature Type Summary
Numerical Features
Categorical Features
Numerical Preprocessing
Categorical Preprocessing
Combined Preprocessing Summary
Model Configuration
Pipeline Summary
Training Summary
Prediction Summary
Prediction Preview
Metrics Table
Classification Report
Confusion Matrix
Confusion Matrix Summary
Model Save Report
Final Model Summary
Validation Checks
Completion Summary
```

---

## 23. Final Validation

The notebook performs final validation checks to confirm that the Model 2 workflow completed successfully.

The checks include:

```text
Training completed
Test predictions generated
Prediction probabilities generated
Accuracy calculated
ROC-AUC calculated
Metrics file saved
Model file saved
```

Each validation check is displayed with either:

```text
PASS
```

or:

```text
CHECK
```

This provides confirmation that the expected training, inference, evaluation and output-generation steps have been completed.

---

## 24. Model 2 Workflow

```text
returns_features.csv
        │
        ▼
Feature / Target Separation
        │
        ▼
80/20 Stratified Train/Test Split
        │
        ▼
Feature Type Identification
        │
        ▼
Preprocessing Pipeline
        │
        ├───────────────────┐
        ▼                   ▼
    Numerical           Categorical
     Pipeline             Pipeline
        │                   │
        ▼                   ▼
Median Imputation   Most-Frequent Imputation
        │                   │
        ▼                   ▼
StandardScaler      One-Hot Encoding
        │                   │
        └─────────┬─────────┘
                  ▼
          XGBoost Classifier
                  │
                  ▼
            Model Training
                  │
                  ▼
             Test Inference
                  │
          ┌───────┴────────┐
          ▼                ▼
   Class Prediction   Return Probability
          │                │
          └───────┬────────┘
                  ▼
        Performance Evaluation
                  │
       ┌──────────┼──────────────┐
       ▼          ▼              ▼
    Metrics   Confusion     Classification
               Matrix          Report
                  │
                  ▼
          Saved Model Results
          │              │
          ▼              ▼
     Metrics CSV     Model Pipeline
```

---

## 25. Model 2 Summary

| Component                     | Implementation                                       |
| ----------------------------- | ---------------------------------------------------- |
| Model                         | XGBoost Classifier                                   |
| Training / Inference Notebook | `notebooks/04_Model2_XGBoost.ipynb`                  |
| Input Dataset                 | `data/processed/returns_features.csv`                |
| Target                        | `Return_Status`                                      |
| Classes                       | `0 = Not Returned`, `1 = Returned`                   |
| Train/Test Split              | 80% / 20%                                            |
| Split Method                  | Stratified                                           |
| Random State                  | 42                                                   |
| Numerical Imputation          | Median                                               |
| Numerical Scaling             | `StandardScaler`                                     |
| Categorical Imputation        | Most Frequent                                        |
| Categorical Encoding          | `OneHotEncoder`                                      |
| Number of Estimators          | 300                                                  |
| Maximum Depth                 | 6                                                    |
| Learning Rate                 | 0.05                                                 |
| Subsample                     | 0.80                                                 |
| Feature Subsampling           | 0.80                                                 |
| Minimum Child Weight          | 2                                                    |
| L2 Regularisation             | 1.0                                                  |
| Objective                     | `binary:logistic`                                    |
| Evaluation Metric             | `logloss`                                            |
| Evaluation                    | Accuracy, Precision, Recall, F1-Score, ROC-AUC       |
| Additional Evaluation         | Classification Report, Confusion Matrix              |
| Model Output                  | `models/model2_xgboost.joblib`                       |
| Metrics Output                | `experimental_results/model2_xgboost_metrics.csv`    |
| Visualisation Output          | `visualisations/model2_xgboost_confusion_matrix.png` |

---

## 26. Relationship to the Project Workflow

Model 2 is the fourth stage of the predictive modelling workflow:

```text
Raw Dataset
    │
    ▼
01_Preprocessing.ipynb
    │
    ▼
returns_preprocessed.csv
    │
    ▼
02_FeatureEngineering.ipynb
    │
    ▼
returns_features.csv
    │
    ▼
04_Model2_XGBoost.ipynb
    │
    ▼
XGBoost Model
    │
    ├── Class Predictions
    ├── Return Probabilities
    ├── Evaluation Metrics
    └── Confusion Matrix
```

Model 2 therefore uses the output generated by the preceding data-preparation and feature-engineering stages.
