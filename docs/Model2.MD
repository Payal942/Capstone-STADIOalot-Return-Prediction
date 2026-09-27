# Model 2 — XGBoost Classifier

## 1. Model 2 Overview

Model 2 is an **XGBoost Classifier** developed to predict whether an STADIOalot e-commerce order will be returned.

The model uses the feature-engineered dataset produced by **Notebook 2 — Feature Engineering**.

The modelling process includes:

* Loading the feature-engineered dataset
* Separating features and target
* Creating an 80/20 stratified train/test split
* Identifying numerical and categorical features
* Applying numerical and categorical preprocessing
* Training the XGBoost classifier
* Generating test-set class predictions and return probabilities
* Evaluating model performance
* Generating a classification report
* Generating a confusion matrix
* Saving the trained model pipeline
* Saving the evaluation metrics
* Performing final validation checks

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

## 2. Model 2 Training and Test Prediction Notebook

The notebook responsible for **Model 2 training and test-set prediction** is:

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

### Test-Set Prediction

After training, the notebook performs prediction on the held-out testing dataset.

It generates:

```text
Class Predictions
Return Probabilities
```

The class prediction identifies whether an order is predicted to be returned, while the probability represents the model's estimated probability for the `Returned` class.

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

The model therefore uses the information contained in the feature-engineered dataset. Post-return information removed during the earlier preprocessing stage is not reintroduced by this notebook.

---

## 4. Why XGBoost?

XGBoost is a gradient-boosting ensemble learning method based on decision trees.

The trees are constructed sequentially, with subsequent trees helping to improve the predictions of previous trees.

XGBoost was selected as Model 2 because it is suitable for structured tabular data and can model complex, non-linear relationships between customer, product, transaction and engineered order characteristics.

Using XGBoost as Model 2 also provides a second tree-based model that can be evaluated using the same feature-engineered dataset and train/test methodology as Model 1.

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
    random_state=RANDOM_STATE,
    stratify=y
)
```

The notebook uses:

```text
TEST_SIZE = 0.20
RANDOM_STATE = 42
```

A stratified split is used so that the proportion of returned and non-returned orders is approximately maintained in both the training and testing datasets.

The fixed `random_state` of `42` makes the split reproducible.

The same 80/20 stratified approach is used for Random Forest and XGBoost to support a consistent model comparison.

---

## 6. Feature Type Identification

The notebook identifies numerical and categorical features using the training feature matrix.

Numerical features are identified using:

```python
X_train.select_dtypes(include=np.number)
```

Categorical features are identified using:

```python
X_train.select_dtypes(exclude=np.number)
```

The notebook displays separate inventories for the numerical and categorical features.

This allows different preprocessing methods to be applied according to feature type.

---

## 7. Data Preprocessing

The XGBoost model uses a scikit-learn `ColumnTransformer` containing separate numerical and categorical preprocessing pipelines.

The preprocessing is included inside the complete modelling pipeline.

This ensures that the preprocessing steps are fitted as part of the training pipeline and then applied consistently when predictions are generated for the testing data.

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
```

and:

```python
StandardScaler()
```

Median imputation replaces missing numerical values using the median calculated during fitting.

`StandardScaler` standardises the numerical feature values.

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
```

and:

```python
OneHotEncoder(handle_unknown="ignore")
```

`handle_unknown="ignore"` ensures that categories not observed during fitting do not cause encoding errors when predictions are generated.

---

## 8. Complete Preprocessing and Model Pipeline

The numerical and categorical preprocessing pipelines are combined using a `ColumnTransformer`.

The resulting preprocessing component is then combined with the XGBoost classifier in a single scikit-learn `Pipeline`.

The overall workflow is:

```text
Input Features
        ↓
ColumnTransformer
        │
        ├───────────────────┐
        ▼                   ▼
    Numerical           Categorical
     Pipeline              Pipeline
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

Keeping preprocessing and modelling inside one pipeline ensures that the same transformations are applied consistently during model training and test-set prediction.

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
    random_state=RANDOM_STATE,
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

`colsample_bytree=0.80` uses 80% of the available input features for each tree.

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

`n_jobs=-1` allows XGBoost to use the available CPU cores for parallel processing.

---

## 10. Model Training

The complete modelling pipeline is trained using:

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

The testing data remains separate until the prediction and evaluation stages.

The notebook displays a training summary containing the model name, number of training rows, number of input features and training status.

---

## 11. Test-Set Prediction

After training, the model generates predictions for the held-out testing dataset:

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

The probability represents the model's estimated probability of the positive class:

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

The ROC-AUC calculation uses the predicted return probabilities generated from the test data.

---

## 14. Classification Report

The notebook generates a classification report using:

```python
classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)
```

The classification report contains:

```text
Precision
Recall
F1-Score
Support
```

The classes are labelled as:

```text
Not Returned
Returned
```

The report also includes:

```text
Accuracy
Macro Average
Weighted Average
```

The classification report is converted into a pandas DataFrame for display and inspection.

The classification report is **not separately saved to a CSV file** by the notebook.

---

## 15. Confusion Matrix

The notebook calculates a confusion matrix using:

```python
cm = confusion_matrix(
    y_test,
    y_pred
)
```

The four outcomes are:

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

The notebook creates a heatmap visualisation of the confusion matrix using Seaborn.

The confusion matrix uses the labels:

```text
Not Returned
Returned
```

---

## 16. Confusion Matrix Visualisation

The notebook saves the XGBoost confusion matrix as:

```text
visualisations/model2_xgboost_confusion_matrix.png
```

The image is generated using:

```python
plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)
```

The visualisation includes:

* Actual class labels on the y-axis
* Predicted class labels on the x-axis
* Confusion-matrix counts
* A heatmap representation of the four classification outcomes

The notebook also displays a summary table containing the True Negative, False Positive, False Negative and True Positive counts.

---

## 17. Evaluation Metrics Output

The notebook calculates the following five evaluation metrics:

```text
accuracy
precision
recall
f1_score
roc_auc
```

These metrics are stored in a pandas DataFrame and saved to:

```text
experimental_results/model2_xgboost_metrics.csv
```

The saved CSV contains the following columns:

```text
model
accuracy
precision
recall
f1_score
roc_auc
```

The model name stored in the output is:

```text
XGBoost
```

The classification report and confusion matrix are displayed during notebook execution but are not separately written to CSV files.

---

## 18. Trained Model Output

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

Saving the complete pipeline preserves the preprocessing and model components together so that the same transformations can be reused when the saved pipeline is loaded for future prediction.

---

## 19. Project Paths and Output Folders

The notebook defines the following project paths:

| Component             | Path                                                 |
| --------------------- | ---------------------------------------------------- |
| Input Dataset         | `data/processed/returns_features.csv`                |
| Model Output          | `models/model2_xgboost.joblib`                       |
| Metrics Output        | `experimental_results/model2_xgboost_metrics.csv`    |
| Visualisations Folder | `visualisations/`                                    |
| Confusion Matrix      | `visualisations/model2_xgboost_confusion_matrix.png` |

The required output directories are created automatically if they do not already exist.

The notebook also displays a path summary containing the resolved full paths.

---

## 20. Notebook Associated With Model 2

The notebook responsible for Model 2 training, test-set prediction and evaluation is:

```text
notebooks/04_Model2_XGBoost.ipynb
```

This is the primary executable file for Model 2.

The complete workflow for Model 2 is contained within this notebook.

---

## 21. Notebook Execution Order

Model 2 depends on the outputs of the preceding preprocessing and feature-engineering notebooks.

The recommended execution order is:

```text
01_Preprocessing.ipynb
        ↓
02_FeatureEngineering.ipynb
        ↓
04_Model2_XGBoost.ipynb
```

### Notebook 1 — Preprocessing

```text
notebooks/01_Preprocessing.ipynb
```

Prepares the cleaned and preprocessed dataset.

The output is:

```text
data/processed/returns_preprocessed.csv
```

### Notebook 2 — Feature Engineering

```text
notebooks/02_FeatureEngineering.ipynb
```

Creates the feature-engineered dataset:

```text
data/processed/returns_features.csv
```

### Notebook 4 — Model 2 XGBoost

```text
notebooks/04_Model2_XGBoost.ipynb
```

Loads the feature-engineered dataset and:

* Separates features and target
* Performs the train/test split
* Identifies feature types
* Applies preprocessing
* Trains XGBoost
* Generates test-set predictions
* Evaluates the model
* Saves the model
* Saves the evaluation metrics
* Saves the confusion matrix visualisation

---

## 22. How to Run Model 2

Model 2 should be executed from the repository root.

### Step 1 — Install Dependencies

Install the project dependencies using the project's requirements file:

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
2. Create the required output folders.
3. Display the path summary.
4. Load `returns_features.csv`.
5. Validate the `Return_Status` target.
6. Separate features and target.
7. Create the 80/20 stratified train/test split.
8. Display the split summary.
9. Verify the target distribution after the split.
10. Identify numerical and categorical features.
11. Configure numerical preprocessing.
12. Configure categorical preprocessing.
13. Combine the preprocessing pipelines.
14. Configure the XGBoost classifier.
15. Build the complete modelling pipeline.
16. Train the XGBoost model.
17. Generate test-set class predictions.
18. Generate return probabilities.
19. Display a prediction preview.
20. Calculate evaluation metrics.
21. Generate the classification report.
22. Generate and display the confusion matrix.
23. Save the confusion matrix visualisation.
24. Save the evaluation metrics.
25. Save the trained model pipeline.
26. Display the model save report.
27. Display the final model summary.
28. Perform final validation checks.
29. Display the Model 2 completion summary.

---

## 23. Model 2 Outputs

After successful execution, the following output files are produced.

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

The notebook also displays the following intermediate and final outputs:

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

Confusion Matrix Visualisation

Model Save Report

Final Model Summary

Validation Checks

Completion Summary
```

---

## 24. Final Model Validation

The notebook performs final validation checks covering:

```text
Training completed

Test predictions generated

Prediction probabilities generated

Accuracy calculated

ROC-AUC calculated

Metrics file saved

Model file saved
```

Each validation check is displayed with a result and a corresponding status:

```text
PASS
```

or:

```text
CHECK
```

The validation output therefore provides a programmatic check of the main training, prediction, evaluation and file-saving steps.

---

## 25. Model 2 Workflow

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
     Pipeline              Pipeline
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
          Test-Set Prediction
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
          Saved Model and Results
             │              │
             ▼              ▼
        Metrics CSV    Model Pipeline
             │
             ▼
       Confusion Matrix PNG
```

---

## 26. Model 2 Summary

| Component                      | Implementation                                       |
| ------------------------------ | ---------------------------------------------------- |
| Model                          | XGBoost Classifier                                   |
| Training / Prediction Notebook | `notebooks/04_Model2_XGBoost.ipynb`                  |
| Input Dataset                  | `data/processed/returns_features.csv`                |
| Target                         | `Return_Status`                                      |
| Classes                        | `0 = Not Returned`, `1 = Returned`                   |
| Train/Test Split               | 80% / 20%                                            |
| Split Method                   | Stratified                                           |
| Random State                   | 42                                                   |
| Numerical Imputation           | Median                                               |
| Numerical Scaling              | `StandardScaler`                                     |
| Categorical Imputation         | Most Frequent                                        |
| Categorical Encoding           | `OneHotEncoder`                                      |
| Unknown Categories             | `handle_unknown="ignore"`                            |
| Number of Estimators           | 300                                                  |
| Maximum Depth                  | 6                                                    |
| Learning Rate                  | 0.05                                                 |
| Subsample                      | 0.80                                                 |
| Feature Subsampling            | 0.80                                                 |
| Minimum Child Weight           | 2                                                    |
| L2 Regularisation              | 1.0                                                  |
| Objective                      | `binary:logistic`                                    |
| Evaluation Metric              | `logloss`                                            |
| Evaluation Outputs             | Accuracy, Precision, Recall, F1-Score, ROC-AUC       |
| Additional Evaluation          | Classification Report, Confusion Matrix              |
| Model Output                   | `models/model2_xgboost.joblib`                       |
| Metrics Output                 | `experimental_results/model2_xgboost_metrics.csv`    |
| Visualisation Output           | `visualisations/model2_xgboost_confusion_matrix.png` |

---

## 27. Relationship to the Project Workflow

Model 2 forms part of the predictive modelling workflow following preprocessing and feature engineering:

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

The resulting trained pipeline, evaluation metrics and confusion matrix visualisation are saved as project outputs for subsequent analysis and comparison with the other predictive model.
