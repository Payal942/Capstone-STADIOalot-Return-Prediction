# CAP182 SS2 – Python Environment

This folder contains the Python environment requirements and experimental configuration used for the **CAP182 SS2 STADIOalot Return Prediction project**.

The files document the software environment, Python packages, and experimental settings used during data preprocessing, feature engineering, model development, and evaluation.

## Python Version

The project was developed and tested using:

**Python 3.14.3**

## Contents

| File               | Description                                                                         |
| ------------------ | ----------------------------------------------------------------------------------- |
| `requirements.txt` | Lists the Python libraries and package versions required for the project.           |
| `configuration.md` | Documents the experimental configuration, modelling settings, and evaluation setup. |

## Python Dependencies

The project uses the following Python libraries:

* **pandas** – Data loading, cleaning, transformation, and analysis
* **NumPy** – Numerical operations and data processing
* **Matplotlib** – Data and model-result visualisation
* **Seaborn** – Statistical visualisation
* **scikit-learn** – Data preprocessing, train/test splitting, Random Forest modelling, evaluation metrics, and statistical comparison
* **XGBoost** – Gradient-boosted decision-tree modelling for return prediction
* **joblib** – Saving and loading trained machine-learning models

The exact package versions used for the project are specified in `requirements.txt`.

## Reproducibility

The purpose of this folder is to document the software environment used during the CAP182 SS2 experiments.

The `requirements.txt` file records the Python packages and versions used during development, while `configuration.md` records the modelling and experimental settings.

This documentation provides the information required to reproduce the project's experimental environment as closely as possible.

## Models

The project includes two machine-learning models:

1. **Random Forest**
2. **XGBoost**

Both models are trained to predict the `Return_Status` target using features available at or before the order-confirmation stage.

## Experimental Configuration

Details regarding data splitting, feature engineering, model parameters, evaluation metrics, and model comparison are documented separately in:

`configuration.md`
