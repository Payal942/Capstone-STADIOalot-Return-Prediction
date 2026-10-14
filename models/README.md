# Models

This folder contains the trained machine learning models developed for the **STADIOalot Return Prediction** project.

The saved models are used to predict whether an e-commerce order is likely to be **Returned** or **Not Returned** based on information available before the return occurs.

## Contents

The folder contains the following trained model files:

| Model                       | File                          | Description                                                                  |
| --------------------------- | ----------------------------- | ---------------------------------------------------------------------------- |
| **Model 1 — Random Forest** | `model1_random_forest.joblib` | Trained Random Forest classification pipeline used to predict order returns. |
| **Model 2 — XGBoost**       | `model2_xgboost.joblib`       | Trained XGBoost classification pipeline used to predict order returns.       |

## Saved Model Pipelines

Both model files are saved using the **Joblib** format (`.joblib`).

The saved pipelines contain the trained machine-learning components required to make predictions from the feature-engineered dataset. This allows the trained models to be loaded and reused without retraining them from scratch.

The pipelines include the relevant preprocessing and model components used during training, including:

* Numerical missing-value handling using median imputation
* Numerical feature scaling using `StandardScaler`
* Categorical missing-value handling using most-frequent imputation
* Categorical encoding using `OneHotEncoder`
* The trained classification model

## Model 1 — Random Forest

**File:**

`models/model1_random_forest.joblib`

The Random Forest model is an ensemble classification algorithm that combines multiple decision trees to predict the target variable `Return_Status`.

The model was trained using the following main hyperparameters:

* `n_estimators = 300`
* `max_depth = 12`
* `min_samples_split = 5`
* `min_samples_leaf = 2`
* `class_weight = "balanced_subsample"`
* `random_state = 42`
* `n_jobs = -1`

## Model 2 — XGBoost

**File:**

`models/model2_xgboost.joblib`

The XGBoost model is a gradient-boosting classification algorithm used to predict the target variable `Return_Status`.

The model was trained using the following main hyperparameters:

* `n_estimators = 300`
* `max_depth = 6`
* `learning_rate = 0.05`
* `subsample = 0.80`
* `colsample_bytree = 0.80`
* `min_child_weight = 2`
* `reg_lambda = 1.0`
* `objective = "binary:logistic"`
* `eval_metric = "logloss"`
* `random_state = 42`
* `n_jobs = -1`

## Target Variable

Both models predict:

**`Return_Status`**

The target variable is encoded as:

* `1` = Returned
* `0` = Not Returned

The models are intended to generate predictions before a return takes place, supporting the project's objective of identifying orders that may be at higher risk of being returned.

## Model Training

The models were trained using the feature-engineered dataset produced by the project preprocessing and feature-engineering stages.

The training process uses an **80/20 stratified train-test split** with `random_state = 42`.

The trained models are saved to this folder after model training so that they can be reused for prediction and evaluation.

## Related Files

The model files work together with other project outputs:

* **Preprocessed dataset:** `data/processed/returns_preprocessed.csv`
* **Feature-engineered dataset:** `data/processed/returns_features.csv`
* **Model 1 metrics:** `experimental_results/model1_random_forest_metrics.csv`
* **Model 2 metrics:** `experimental_results/model2_xgboost_metrics.csv`
* **Model 1 confusion matrix:** `visualisations/model1_random_forest_confusion_matrix.png`
* **Model 2 confusion matrix:** `visualisations/model2_xgboost_confusion_matrix.png`

## Reproducibility

The saved `.joblib` files are generated from the model-training notebooks:

* `notebooks/03_Model1_RandomForest.ipynb`
* `notebooks/04_Model2_XGBoost.ipynb`

Running these notebooks with the required project dependencies and datasets will reproduce the model-training process and regenerate the saved model artefacts.



---

# Client-Data Model Artefacts

The `SS3_Python.ipynb` notebook generates two additional model artefacts when training the XGBoost return-prediction model using the supplied STADIOalot client data.

## Additional Model Files

| File | Description |
|---|---|
| `ss3_xgboost_return_prediction.joblib` | The trained XGBoost classification model used to predict whether an order will be returned. |
| `ss3_preprocessor.joblib` | The fitted preprocessing component used to transform the client data into the format expected by the trained model. |

### 1. Trained XGBoost Model

**File path:**

```text
models/ss3_xgboost_return_prediction.joblib
```

This file stores the trained XGBoost classifier developed using the integrated client dataset. It is intended to predict the `Return_Status` target variable:

- `0` = Not Returned
- `1` = Returned

The model supports return-risk prediction using information available at or before the intended prediction point.

### 2. Fitted Preprocessing Pipeline

**File path:**

```text
models/ss3_preprocessor.joblib
```

This file stores the fitted preprocessing component used to prepare the client dataset for model input. Depending on the transformations defined in the notebook, this may include handling missing values and encoding categorical features.

Saving the preprocessor separately allows the same fitted transformations to be reused when preparing new data for prediction, helping maintain consistency between training and inference.

### Model Loading and Reuse

The saved artefacts can be loaded using Joblib:

```python
import joblib

model = joblib.load(
    "models/ss3_xgboost_return_prediction.joblib"
)

preprocessor = joblib.load(
    "models/ss3_preprocessor.joblib"
)
```

The preprocessing component and trained model should be used with the same feature structure and transformations expected during training. The exact prediction workflow should follow the implementation in `SS3_Python.ipynb`.
