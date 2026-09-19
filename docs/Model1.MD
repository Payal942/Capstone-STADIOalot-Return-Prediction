# Model 1 — Random Forest Classifier

## 1. Model 1 Overview

Model 1 is a **Random Forest Classifier** developed to predict whether an STADIOalot e-commerce order will be returned.

The model uses the feature-engineered dataset produced by **Notebook 2 — Feature Engineering**. The modelling process includes:

* Loading the feature-engineered dataset
* Separating features and target
* Creating an 80/20 stratified train/test split
* Identifying numerical and categorical features
* Applying preprocessing
* Training the Random Forest classifier
* Generating class predictions and return probabilities
* Evaluating model performance
* Saving the trained model
* Saving the evaluation metrics
* Generating and saving a confusion matrix

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

## 2. Model 1 Training and Inference File

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
80/20 Stratified Split
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

After training, the notebook performs inference on the unseen testing dataset.

It generates:

```text
Class Predictions
Return Probabilities
```

The class prediction identifies whether the order is predicted to be returned, while the probability represents the estimated probability of the `Returned` class.

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

The model target is:

```text
Return_Status
```

The feature matrix is created by removing the target:

```python
X = df.drop(columns=["Return_Status"])
```

The target is converted to integer values:

```python
y = df["Return_Status"].astype(int)
```

The remaining feature columns are used as predictor variables.

---

## 4. Why Random Forest?

Random Forest is an ensemble learning method that combines multiple decision trees to perform classification.

It was selected as Model 1 because e-commerce return behaviour can involve non-linear relationships between customer, product, transaction and engineered order characteristics.

Random Forest also provides a suitable tree-based baseline for comparison with the second model, XGBoost.

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

A stratified split is used so that the proportion of returned and non-returned orders is maintained across the training and testing datasets.

A fixed `random_state` of `42` makes the split reproducible.

The same 80/20 stratified methodology is used for Model 2 to support consistent model comparison.

---

## 6. Feature Identification

The notebook identifies numerical and categorical features using the training dataset.

Numerical features are identified using:

```python
X_train.select_dtypes(include=np.number)
```

Categorical features are identified using:

```python
X_train.select_dtypes(exclude=np.number)
```

This allows different preprocessing methods to be applied to each feature type.

---

## 7. Data Preprocessing

The preprocessing is implemented using a scikit-learn `ColumnTransformer`.

The preprocessing is part of the complete modelling pipeline, ensuring that the transformations used during training are also applied consistently during inference.

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
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)
```

Median imputation replaces missing numerical values using the median calculated from the training data.

`StandardScaler` standardises numerical variables.

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
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)
```

`handle_unknown="ignore"` prevents errors when an unseen category is encountered during inference.

---

## 8. Random Forest Configuration

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
    random_state=42,
    n_jobs=-1
)
```

### Parameter Description

* **`n_estimators=300`** — uses 300 decision trees.
* **`max_depth=12`** — limits the maximum depth of each tree.
* **`min_samples_split=5`** — requires at least five observations for a node to be split.
* **`min_samples_leaf=2`** — requires at least two observations in each terminal leaf.
* **`class_weight="balanced_subsample"`** — adjusts class weighting for each bootstrap sample.
* **`random_state=42`** — supports reproducibility.
* **`n_jobs=-1`** — allows all available CPU cores to be used.

---

## 9. Complete Model Pipeline

The preprocessing and Random Forest classifier are combined into one scikit-learn pipeline.

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
        ├───────────────┐
        ▼               ▼
 Numerical          Categorical
 Features            Features
        │               │
        ▼               ▼
Median Imputation  Most-Frequent
        │           Imputation
        ▼               │
StandardScaler          ▼
        │          One-Hot Encoding
        └───────┬───────┘
                ▼
        Random Forest
          Classifier
                │
                ▼
          Trained Pipeline
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
            random_forest
        )
    ]
)
```

This approach ensures that preprocessing is fitted using the training data and then consistently applied when predictions are generated.

---

## 10. Model Training

The complete pipeline is trained using:

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
Preprocessing
        ↓
Random Forest Training
        ↓
Trained Model Pipeline
```

The notebook confirms that training has completed before proceeding to inference.

---

## 11. Inference

After training, predictions are generated using the unseen test dataset.

### Class Prediction

The predicted class is generated using:

```python
y_pred = model_pipeline.predict(X_test)
```

The output represents:

```text
0 = Not Returned
1 = Returned
```

### Return Probability

The probability of the `Returned` class is generated using:

```python
y_probability = (
    model_pipeline
    .predict_proba(X_test)[:, 1]
)
```

This probability is used for ROC-AUC evaluation.

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

The first ten test-set predictions are displayed for inspection.

This allows the actual outcome, predicted outcome and estimated return probability to be examined together.

---

## 13. Model Evaluation

Model 1 is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* Classification Report

### Accuracy

Measures the proportion of test observations classified correctly.

### Precision

Measures the proportion of observations predicted as `Returned` that were actually returned.

### Recall

Measures the proportion of actually returned orders that were correctly identified.

### F1-Score

Provides a combined measure of precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between returned and non-returned orders across classification thresholds.

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

---

## 15. Confusion Matrix

A confusion matrix is generated to examine the classification results.

The four outcomes are:

| Outcome        | Meaning                                      |
| -------------- | -------------------------------------------- |
| True Negative  | Correctly predicted Not Returned             |
| False Positive | Predicted Returned but actually Not Returned |
| False Negative | Predicted Not Returned but actually Returned |
| True Positive  | Correctly predicted Returned                 |

The notebook generates a labelled confusion-matrix heatmap.

The saved visualisation is:

```text
visualisations/model1_random_forest_confusion_matrix.png
```

---

## 16. Model Outputs

### Trained Model

The complete preprocessing and Random Forest pipeline is saved to:

```text
models/model1_random_forest.joblib
```

### Evaluation Metrics

The evaluation metrics are saved to:

```text
experimental_results/model1_random_forest_metrics.csv
```

The metrics file contains:

```text
model
accuracy
precision
recall
f1_score
roc_auc
```

### Confusion Matrix

The confusion matrix is saved to:

```text
visualisations/model1_random_forest_confusion_matrix.png
```

---

## 17. How to Run Model 1

Model 1 depends on the output produced by the previous notebooks.

Run the notebooks in this order:

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

1. Define the project paths.
2. Load `returns_features.csv`.
3. Validate `Return_Status`.
4. Separate features and target.
5. Create the 80/20 stratified split.
6. Verify the target distribution.
7. Identify numerical and categorical features.
8. Configure numerical preprocessing.
9. Configure categorical preprocessing.
10. Combine the preprocessing pipelines.
11. Configure the Random Forest classifier.
12. Build the complete modelling pipeline.
13. Train the model.
14. Generate test-set predictions.
15. Generate return probabilities.
16. Display a prediction preview.
17. Calculate evaluation metrics.
18. Generate the classification report.
19. Generate the confusion matrix.
20. Save the evaluation metrics.
21. Save the trained model.
22. Perform final validation checks.
23. Display the Model 1 completion summary.

---

## 18. Model 1 Notebook Outputs

Successful execution produces the following files:

```text
models/
└── model1_random_forest.joblib

experimental_results/
└── model1_random_forest_metrics.csv

visualisations/
└── model1_random_forest_confusion_matrix.png
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
Numerical Preprocessing Summary
Categorical Preprocessing Summary
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

## 19. Final Validation

The notebook performs validation checks before completion.

The checks confirm that:

```text
Training completed
Test predictions generated
Prediction probabilities generated
Accuracy calculated
ROC-AUC calculated
Metrics file saved
Model file saved
```

Each check is displayed with either:

```text
PASS
```

or:

```text
CHECK
```

This provides a final confirmation that the Model 1 training and output process completed successfully.

---

## 20. Model 1 Workflow

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
        │                   │
        ▼                   ▼
Median Imputation   Most-Frequent Imputation
        │                   │
        ▼                   ▼
StandardScaler      One-Hot Encoding
        │                   │
        └─────────┬─────────┘
                  ▼
        Random Forest Classifier
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
       ┌──────────┼──────────┐
       ▼          ▼          ▼
    Metrics   Confusion   Classification
               Matrix        Report
                  │
                  ▼
          Saved Model Results
          │              │
          ▼              ▼
     Metrics CSV     Model Pipeline
```

---

## 21. Model 1 Summary

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
| Number of Trees               | 300                                                        |
| Maximum Depth                 | 12                                                         |
| Minimum Samples Split         | 5                                                          |
| Minimum Samples Leaf          | 2                                                          |
| Class Weight                  | `balanced_subsample`                                       |
| Evaluation                    | Accuracy, Precision, Recall, F1-Score, ROC-AUC             |
| Additional Evaluation         | Classification Report, Confusion Matrix                    |
| Model Output                  | `models/model1_random_forest.joblib`                       |
| Metrics Output                | `experimental_results/model1_random_forest_metrics.csv`    |
| Visualisation Output          | `visualisations/model1_random_forest_confusion_matrix.png` |

---

## 22. Relationship to the Project Workflow

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
Random Forest Model
    │
    ├── Predictions
    ├── Evaluation Metrics
    └── Confusion Matrix
```

Model 1 therefore depends on the processed and feature-engineered outputs produced by the preceding notebooks.
