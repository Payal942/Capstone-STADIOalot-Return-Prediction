# Model 1 — Random Forest Classifier

## 1. Model 1 Overview

Model 1 is a **Random Forest Classifier** developed to predict whether an STADIOalot e-commerce order will be returned.

The model uses the feature-engineered dataset produced by **Notebook 2 — Feature Engineering**. The modelling process includes:

* Loading the feature-engineered dataset
* Separating the features and target variable
* Creating an 80/20 stratified train/test split
* Identifying numerical and categorical features
* Applying numerical and categorical preprocessing
* Configuring the Random Forest classifier
* Building a complete Scikit-learn modelling pipeline
* Training the Random Forest model
* Generating class predictions
* Generating return probabilities
* Evaluating model performance
* Generating a classification report
* Generating a confusion matrix
* Saving the trained model pipeline
* Saving the evaluation metrics
* Saving the confusion matrix visualisation
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

## 2. Model 1 Training and Inference Notebook

The notebook responsible for Model 1 training and inference is:

```text
notebooks/03_Model1_RandomForest.ipynb
```

This notebook contains the complete implementation for Model 1.

### Training

The notebook trains the Random Forest model using the training portion of the feature-engineered dataset:

```text
data/processed/returns_features.csv
        ↓
Feature / Target Separation
        ↓
80/20 Stratified Train/Test Split
        ↓
Training Data
        ↓
Preprocessing
        ↓
Random Forest Classifier
        ↓
Trained Model Pipeline
```

### Inference

After training, inference is performed on the unseen testing dataset.

The notebook generates:

```text
Class Predictions
Return Probabilities
```

The class prediction identifies whether an order is predicted as returned or not returned.

The return probability is obtained using the probability associated with class `1`, representing the predicted probability of the **Returned** class.

---

## 3. Model Input

Model 1 uses the feature-engineered dataset:

```text
data/processed/returns_features.csv
```

This dataset is created by:

```text
notebooks/02_FeatureEngineering.ipynb
```

The target variable is:

```text
Return_Status
```

The feature matrix is created by removing the target:

```python
X = df.drop(
    columns=["Return_Status"]
)
```

The target is converted to integer values:

```python
y = df["Return_Status"].astype(int)
```

The remaining columns in `X` are used as predictor variables.

The Model 1 notebook therefore expects the input dataset to have already undergone the preprocessing and feature-engineering stages performed by the preceding notebooks.

---

## 4. Why Random Forest?

Random Forest is an ensemble learning method that combines multiple decision trees to perform classification.

It was selected as Model 1 because e-commerce return behaviour may involve non-linear relationships between customer, product, transaction and engineered order characteristics.

Random Forest also provides a tree-based model for comparison with **Model 2 — XGBoost**.

---

## 5. Train/Test Split

The feature-engineered dataset is divided into:

```text
80% Training Data
20% Testing Data
```

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

A **stratified split** is used so that the distribution of the target classes is maintained across the training and testing datasets.

The following configuration is used:

| Parameter      | Value |
| -------------- | ----: |
| `test_size`    |  0.20 |
| `random_state` |    42 |
| `stratify`     |   `y` |

The fixed `random_state=42` makes the split reproducible.

The notebook also produces a split summary showing the number of observations and percentage assigned to the full dataset, training set and testing set.

A target-distribution report is also generated to display the percentage of `Not Returned` and `Returned` observations in the training and testing datasets.

The same 80/20 stratified methodology is used for Model 2 to support consistent model comparison.

---

## 6. Feature Identification

Numerical and categorical features are identified using the **training dataset**.

Numerical features are identified using:

```python
X_train.select_dtypes(include=np.number)
```

Categorical features are identified using:

```python
X_train.select_dtypes(exclude=np.number)
```

This allows separate preprocessing pipelines to be applied according to feature type.

The notebook generates:

* A feature-type summary
* A numerical feature table
* A categorical feature table

The feature-type summary records the number of numerical, categorical and total input features.

---

## 7. Data Preprocessing

The preprocessing is implemented using a Scikit-learn `ColumnTransformer`.

The numerical and categorical preprocessing pipelines are combined before being passed to the Random Forest classifier.

The preprocessing is included inside the complete modelling pipeline so that the transformations fitted during training are retained and reused during inference.

### 7.1 Numerical Features

Numerical features are processed using:

```text
Median Imputation
        ↓
StandardScaler
```

The implementation is:

```python
numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)
```

### Median Imputation

`SimpleImputer(strategy="median")` replaces missing numerical values using the median calculated from the training data.

### StandardScaler

`StandardScaler` standardises numerical variables before they are passed to the Random Forest classifier.

Although scaling is not generally required by tree-based models, it is retained here as part of the defined modelling preprocessing pipeline.

---

### 7.2 Categorical Features

Categorical features are processed using:

```text
Most-Frequent Imputation
        ↓
One-Hot Encoding
```

The implementation is:

```python
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)
```

### Most-Frequent Imputation

`SimpleImputer(strategy="most_frequent")` replaces missing categorical values with the most frequently occurring category calculated from the training data.

### One-Hot Encoding

`OneHotEncoder` converts categorical variables into numerical indicator features.

The configuration:

```python
handle_unknown="ignore"
```

allows the pipeline to handle categories that were not observed during training without raising an encoding error during inference.

---

## 8. Combined Preprocessing

The numerical and categorical pipelines are combined using a `ColumnTransformer`:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)
```

The resulting preprocessing structure is:

```text
Input Features
       │
       ├─────────────────────┐
       ▼                     ▼
Numerical Features      Categorical Features
       │                     │
       ▼                     ▼
Median Imputation      Most-Frequent Imputation
       │                     │
       ▼                     ▼
StandardScaler         One-Hot Encoding
       │                     │
       └──────────┬──────────┘
                  ▼
       Combined Preprocessor
```

The notebook also displays a combined preprocessing summary showing:

* Feature type
* Imputation method
* Transformation method
* Unknown-category handling

---

## 9. Random Forest Configuration

The Random Forest classifier is configured as follows:

| Parameter           |                Value |
| ------------------- | -------------------: |
| `n_estimators`      |                  300 |
| `max_depth`         |                   12 |
| `min_samples_split` |                    5 |
| `min_samples_leaf`  |                    2 |
| `class_weight`      | `balanced_subsample` |
| `random_state`      |                   42 |
| `n_jobs`            |                   -1 |

The implementation is:

```python
random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
```

### Parameter Description

* **`n_estimators=300`** — creates 300 decision trees in the Random Forest.
* **`max_depth=12`** — limits the maximum depth of each decision tree.
* **`min_samples_split=5`** — requires at least five samples in a node before it can be split.
* **`min_samples_leaf=2`** — requires at least two samples in each terminal leaf.
* **`class_weight="balanced_subsample"`** — adjusts class weights separately for each bootstrap sample.
* **`random_state=42`** — supports reproducible model training.
* **`n_jobs=-1`** — allows the model to use all available CPU cores.

---

## 10. Complete Model Pipeline

The preprocessing and Random Forest classifier are combined into a single Scikit-learn `Pipeline`.

The implementation is:

```python
model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            random_forest
        )
    ]
)
```

The complete modelling structure is:

```text
Feature-Engineered Dataset
        │
        ▼
Feature / Target Separation
        │
        ▼
80/20 Stratified Split
        │
        ▼
ColumnTransformer
        │
        ├────────────────────┐
        ▼                    ▼
Numerical              Categorical
Features               Features
        │                    │
        ▼                    ▼
Median Imputation     Most-Frequent Imputation
        │                    │
        ▼                    ▼
StandardScaler        One-Hot Encoding
        │                    │
        └──────────┬─────────┘
                   ▼
          Random Forest Classifier
                   │
                   ▼
          Trained Model Pipeline
```

This structure ensures that preprocessing is fitted as part of the training pipeline and subsequently reused when predictions are generated.

The complete pipeline is also the object saved to the Model 1 `.joblib` file.

---

## 11. Model Training

The complete modelling pipeline is trained using:

```python
model_pipeline.fit(
    X_train,
    y_train
)
```

The training process is:

```text
X_train + y_train
        ↓
Numerical / Categorical Preprocessing
        ↓
Random Forest Training
        ↓
Trained Model Pipeline
```

The notebook generates a training summary containing:

* Model
* Number of training rows
* Number of training features
* Training status

The training status is reported as:

```text
COMPLETED
```

---

## 12. Inference

After training, predictions are generated using the unseen test dataset.

### 12.1 Class Prediction

The predicted class is generated using:

```python
y_pred = model_pipeline.predict(
    X_test
)
```

The predicted classes represent:

```text
0 = Not Returned
1 = Returned
```

### 12.2 Return Probability

The probability of the `Returned` class is generated using:

```python
y_probability = (
    model_pipeline
    .predict_proba(X_test)[:, 1]
)
```

The `[:, 1]` selection retrieves the predicted probability associated with class `1`, representing the `Returned` class.

These probabilities are subsequently used for ROC-AUC evaluation.

The notebook creates a prediction summary confirming that both class predictions and return probabilities were generated for the test observations.

---

## 13. Prediction Preview

The notebook creates a prediction preview containing:

```text
Actual Return Status
Actual Label
Predicted Return Status
Predicted Label
Return Probability
```

The first ten test-set predictions are displayed.

The preview allows the actual outcome, predicted outcome and estimated return probability to be examined together.

The labels are mapped as:

```text
0 → Not Returned
1 → Returned
```

---

## 14. Model Evaluation

Model 1 is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Classification Report
* Confusion Matrix

### Accuracy

Accuracy measures the proportion of test observations that were classified correctly.

It is calculated using:

```python
accuracy_score(
    y_test,
    y_pred
)
```

### Precision

Precision measures the proportion of observations predicted as `Returned` that were actually returned.

It is calculated using:

```python
precision_score(
    y_test,
    y_pred,
    zero_division=0
)
```

### Recall

Recall measures the proportion of actually returned orders that were correctly identified.

It is calculated using:

```python
recall_score(
    y_test,
    y_pred,
    zero_division=0
)
```

### F1-Score

F1-Score combines precision and recall into a single metric.

It is calculated using:

```python
f1_score(
    y_test,
    y_pred,
    zero_division=0
)
```

### ROC-AUC

ROC-AUC evaluates the model's ability to distinguish between the two classes across classification thresholds.

The calculation uses the predicted return probabilities:

```python
roc_auc_score(
    y_test,
    y_probability
)
```

The notebook presents the five primary metric values in a metrics table.

---

## 15. Classification Report

The notebook generates a Scikit-learn classification report containing:

```text
Precision
Recall
F1-Score
Support
```

The class labels are displayed as:

```text
Not Returned
Returned
```

The report also contains:

```text
Accuracy
Macro Average
Weighted Average
```

The classification report is converted into a Pandas DataFrame for display.

---

## 16. Confusion Matrix

A confusion matrix is generated using:

```python
cm = confusion_matrix(
    y_test,
    y_pred
)
```

The confusion matrix uses the following class labels:

```text
Not Returned
Returned
```

The four outcomes are:

| Outcome        | Meaning                          |
| -------------- | -------------------------------- |
| True Negative  | Correctly predicted Not Returned |
| False Positive | Incorrectly predicted Returned   |
| False Negative | Missed a Returned order          |
| True Positive  | Correctly predicted Returned     |

The notebook generates a labelled heatmap using Seaborn.

The visualisation is titled:

```text
Confusion Matrix – STADIOalot Return Prediction
```

The generated image is saved to:

```text
visualisations/model1_random_forest_confusion_matrix.png
```

The image is saved at 300 DPI.

---

## 17. Confusion Matrix Summary

After calculating the confusion matrix, the notebook extracts:

```python
tn, fp, fn, tp = cm.ravel()
```

A summary table is then created containing:

* True Negatives
* False Positives
* False Negatives
* True Positives

The table also provides the meaning of each outcome.

The notebook additionally verifies whether the confusion matrix image was successfully saved.

---

## 18. Model Outputs

### 18.1 Trained Model

The complete preprocessing and Random Forest pipeline is saved using `joblib` to:

```text
models/model1_random_forest.joblib
```

The saved object contains the complete modelling pipeline, including:

```text
Preprocessing
+
Random Forest Classifier
```

This allows the same preprocessing steps to be applied when the saved model is later used for inference.

### 18.2 Evaluation Metrics

The evaluation metrics are saved to:

```text
experimental_results/model1_random_forest_metrics.csv
```

The saved CSV contains the following fields:

```text
model
accuracy
precision
recall
f1_score
roc_auc
```

### 18.3 Confusion Matrix

The confusion matrix visualisation is saved to:

```text
visualisations/model1_random_forest_confusion_matrix.png
```

---

## 19. How to Run Model 1

Model 1 depends on the outputs produced by the previous notebooks.

Run the notebooks in the following order:

```text
01_Preprocessing.ipynb
        ↓
02_FeatureEngineering.ipynb
        ↓
03_Model1_RandomForest.ipynb
```

### Step 1 — Install Dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

### Step 2 — Start Jupyter

```bash
jupyter notebook
```

### Step 3 — Run the Required Previous Notebooks

Run:

```text
notebooks/01_Preprocessing.ipynb
```

followed by:

```text
notebooks/02_FeatureEngineering.ipynb
```

Notebook 2 must successfully create:

```text
data/processed/returns_features.csv
```

### Step 4 — Open Model 1

Open:

```text
notebooks/03_Model1_RandomForest.ipynb
```

### Step 5 — Execute the Notebook

Run the cells from top to bottom.

The notebook will:

1. Define the project root and file paths.
2. Create the required output directories.
3. Display the path summary.
4. Load `returns_features.csv`.
5. Validate that `Return_Status` exists.
6. Separate features and target.
7. Create the 80/20 stratified train/test split.
8. Display the split summary.
9. Verify the target distribution across the split.
10. Identify numerical features.
11. Identify categorical features.
12. Configure numerical preprocessing.
13. Configure categorical preprocessing.
14. Combine the preprocessing pipelines.
15. Configure the Random Forest classifier.
16. Build the complete modelling pipeline.
17. Train the model.
18. Generate test-set class predictions.
19. Generate return probabilities.
20. Display a prediction preview.
21. Calculate evaluation metrics.
22. Generate the classification report.
23. Generate the confusion matrix.
24. Generate the confusion matrix summary.
25. Save the evaluation metrics.
26. Save the trained model pipeline.
27. Display the final model summary.
28. Perform final validation checks.
29. Display the Model 1 completion summary.

---

## 20. Model 1 Notebook Outputs

Successful execution produces the following files:

```text
models/
└── model1_random_forest.joblib

experimental_results/
└── model1_random_forest_metrics.csv

visualisations/
└── model1_random_forest_confusion_matrix.png
```

The notebook also displays the following summaries and reports:

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
Metrics Save Report
Model Save Report
Final Model Summary
Validation Checks
Model 1 Completion Summary
```

---

## 21. Final Model Validation

The notebook performs final validation checks before completion.

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

Each check produces a Boolean result and is converted into one of the following statuses:

```text
PASS
CHECK
```

The validation logic checks that:

* The trained pipeline has completed fitting.
* The number of predictions matches the number of test observations.
* The number of prediction probabilities matches the number of test observations.
* Accuracy has been calculated and is not `NaN`.
* ROC-AUC has been calculated and is not `NaN`.
* The metrics file exists.
* The trained model file exists.

The validation stage provides a final check that the Model 1 training, inference and output-saving process completed as expected.

---

## 22. Model 1 Workflow

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
        ├─────────────────────┐
        ▼                     ▼
    Numerical             Categorical
     Features               Features
        │                     │
        ▼                     ▼
Median Imputation       Most-Frequent Imputation
        │                     │
        ▼                     ▼
StandardScaler          One-Hot Encoding
        │                     │
        └──────────┬──────────┘
                   ▼
          Random Forest Classifier
                   │
                   ▼
             Model Training
                   │
                   ▼
             Test Inference
                   │
          ┌────────┴────────┐
          ▼                 ▼
   Class Prediction   Return Probability
          │                 │
          └────────┬────────┘
                   ▼
          Performance Evaluation
                   │
       ┌───────────┼────────────┐
       ▼           ▼            ▼
    Metrics    Confusion    Classification
                Matrix          Report
       │
       ▼
  Saved Model Results
       │
       ├───────────────────┐
       ▼                   ▼
   Metrics CSV       Model Pipeline
                           │
                           ▼
                 Confusion Matrix PNG
```

---

## 23. Model 1 Summary

| Component                     | Implementation                                             |
| ----------------------------- | ---------------------------------------------------------- |
| Model                         | Random Forest Classifier                                   |
| Training / Inference Notebook | `notebooks/03_Model1_RandomForest.ipynb`                   |
| Input Dataset                 | `data/processed/returns_features.csv`                      |
| Target                        | `Return_Status`                                            |
| Classes                       | `0 = Not Returned`, `1 = Returned`                         |
| Train/Test Split              | 80% / 20%                                                  |
| Split Method                  | Stratified                                                 |
| Random State                  | 42                                                         |
| Numerical Imputation          | Median                                                     |
| Numerical Scaling             | `StandardScaler`                                           |
| Categorical Imputation        | Most Frequent                                              |
| Categorical Encoding          | `OneHotEncoder`                                            |
| Unknown Categories            | `handle_unknown="ignore"`                                  |
| Number of Trees               | 300                                                        |
| Maximum Depth                 | 12                                                         |
| Minimum Samples Split         | 5                                                          |
| Minimum Samples Leaf          | 2                                                          |
| Class Weight                  | `balanced_subsample`                                       |
| Number of Jobs                | -1                                                         |
| Evaluation Metrics            | Accuracy, Precision, Recall, F1-Score, ROC-AUC             |
| Additional Evaluation         | Classification Report, Confusion Matrix                    |
| Model Output                  | `models/model1_random_forest.joblib`                       |
| Metrics Output                | `experimental_results/model1_random_forest_metrics.csv`    |
| Visualisation Output          | `visualisations/model1_random_forest_confusion_matrix.png` |

---

## 24. Relationship to the Project Workflow

Model 1 is the third stage of the predictive modelling workflow:

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
03_Model1_RandomForest.ipynb
    │
    ▼
Random Forest Pipeline
    │
    ├── Class Predictions
    ├── Return Probabilities
    ├── Evaluation Metrics
    └── Confusion Matrix
```

Model 1 therefore depends on the processed and feature-engineered output produced by the preceding notebooks.

The resulting Random Forest pipeline, evaluation metrics and confusion matrix are saved as project outputs for subsequent analysis and comparison with Model 2.
