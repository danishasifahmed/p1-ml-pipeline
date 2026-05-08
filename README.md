# End-to-End ML Pipeline with MLflow

![Python](https://img.shields.io/badge/Python-3.10-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## What This Project Demonstrates

A complete, production-quality ML pipeline built from scratch on the
California Housing dataset. This project covers every stage a real
MLOps team handles: data ingestion, cleaning, feature engineering,
experiment tracking, model evaluation, and production-style prediction.

## Pipeline Architecture

| Stage | Script | What happens |
|---|---|---|
| 1. Raw Data | sklearn dataset | California Housing loaded directly — no download needed |
| 2. Data Prep | `data_prep.ipynb` | Outlier removal (IQR), train/test split, StandardScaler fitted on train only |
| 3. Feature Engineering | `feature_engineering.ipynb` | 6 new features: RoomsPerBedroom, IncomePerOccupant, IsNearCoast, IsMajorMetro, IncomeAgeInteraction, PopulationPerRoom |
| 4. Training + MLflow | `train.ipynb` | LinearRegression, Ridge, RandomForest trained and logged — params, metrics, artifacts all recorded |
| 5. Evaluation | `evaluate.ipynb` | Residual analysis, predicted vs actual, error by price band, best model registered in MLflow Model Registry |
| 6. Prediction | `predict.ipynb` | Production-style predictor class — single + batch prediction, input validation, confidence intervals, JSON API simulation |

## Results

| Model | Test R² | Test RMSE | Test MAE |
|---|---|---|---|
| Linear Regression | ~0.62 | ~0.60 | ~0.43 |
| Ridge Regression | ~0.62 | ~0.60 | ~0.43 |
| **Random Forest** | **~0.83** | **~0.40** | **~0.28** |

**Best model: Random Forest** — explains ~83% of house price
variance with an average prediction error of ~$28,000.

---

## Key Engineering Decisions

**1. Scaler fitted on training data only**
The StandardScaler is fitted exclusively on training data and
then applied (not re-fitted) to test and prediction data.
This prevents data leakage — a common mistake that inflates
evaluation scores in development but causes failures in production.

**2. Feature engineering justified by domain knowledge**
Every engineered feature has a real-world reason. IncomePerOccupant
captures wealth-per-person better than raw income. IsNearCoast
captures California's coastal price premium. Features without
logical justification were not added.

**3. Preprocessing saved as a Pipeline artifact**
The preprocessor is saved alongside the model so that prediction
always uses the exact same transformation learned during training.
This eliminates an entire class of production bugs.

**4. Confidence intervals on predictions**
Predictions include 95% confidence intervals derived from the
spread of individual tree predictions. This gives stakeholders
a range rather than a false single-point certainty.

**5. Input validation before prediction**
The prediction script validates all inputs for missing fields,
null values, and out-of-range values before any computation.
Errors are returned as clear messages, not silent wrong predictions.

---

## Project Structure
p1-ml-pipeline/
├── notebooks/
│    p1-exploration.ipynb      # Day 1: EDA and dataset exploration

│    data_prep.ipynb           # Day 2: Cleaning, scaling, splitting

│    feature_engineering.ipynb # Day 3: Feature creation and pipeline

│    train.ipynb               # Day 4: Model training + MLflow tracking

│    evaluate.ipynb            # Day 5: Evaluation + MLflow artifacts

│    predict.ipynb             # Day 6: Production prediction script

├── data/

│    raw/                      # Raw dataset and EDA plots

├── models/                       # Saved model and preprocessor files

├── requirements.txt

└── README.md

---

## MLflow Experiment Tracking

All training runs are logged to MLflow with full reproducibility:
- **Parameters**: model type, hyperparameters, feature counts
- **Metrics**: RMSE, MAE, R² for both train and test sets
- **Artifacts**: trained model, feature importances, evaluation plots
- **Model Registry**: best model registered as `CaliforniaHousing-BestModel`

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/danishasifahmed/p1-ml-pipeline.git
cd p1-ml-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open any notebook in Google Colab or Jupyter
# Run notebooks in order: exploration → data_prep → feature_engineering
#                         → train → evaluate → predict
```

---

## Sample Prediction

```python
predictor = HousePricePredictor()

result = predictor.predict_single({
    "MedInc":     8.5,
    "HouseAge":   20.0,
    "AveRooms":   6.5,
    "AveBedrms":  1.2,
    "Population": 1200.0,
    "AveOccup":   2.8,
    "Latitude":   37.77,
    "Longitude":  -122.42
})

# Output:
# Predicted price:  $385,000
# 95% CI:          $310,000 — $460,000
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| scikit-learn | ML models and preprocessing |
| MLflow | Experiment tracking and model registry |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualisation |
| Google Colab | Development environment |

---

## About

Built as Project by
**Danish Asif Ahmed** — ML/AI Engineer targeting MLOps
roles.

[GitHub](https://github.com/danishasifahmed) •
[LinkedIn](https://www.linkedin.com/in/danish-asif-ahmed-968650229/)
