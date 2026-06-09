# `design.md` — ML Project Architecture & Pipeline

**Project:** Multiple Linear Regression on the `50_Startups` dataset  
**Version:** 1.0  
**Role:** Single Source of Truth for workspace organization, pipeline stages, and technology decisions.

---

## 1. Technology Stack

| Layer              | Tool / Library                | Version  |
|--------------------|-------------------------------|----------|
| Language           | Python                        | ≥ 3.10   |
| Data Manipulation  | pandas                        | ≥ 2.0    |
| Numerical Comput.  | numpy                         | ≥ 1.24   |
| Preprocessing      | scikit-learn ≥ 1.3            |          |
| Modelling          | scikit-learn                  |          |
| Metrics            | scikit-learn (r2_score, etc.) |          |
| Serialization      | joblib / pickle               | stdlib   |
| Environment        | pip + requirements.txt        |          |

---

## 2. Directory Tree

```
project-root/
│
├── design.md                          # ← THIS FILE (single source of truth)
├── requirements.txt                   # Pinned dependencies
├── main.py                            # Pipeline orchestrator (runs stages in order)
│
├── data/
│   ├── raw/                           # READ-ONLY  — immutable source data
│   │   └── 50_startups.csv
│   └── processed/                     # WRITE-ONLY — output of data_prep stage
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── src/
│   ├── __init__.py
│   ├── data_display.py                # STAGE 1 — Read-only basic info & EDA
│   ├── data_prep.py                   # STAGE 2 — Consolidated data preparation
│   ├── model_train.py                 # STAGE 3 — Model training
│   └── model_eval.py                  # STAGE 4 — Model evaluation
│
├── notebooks/
│   └── 01_eda.ipynb                   # Optional interactive exploration (read-only)
│
└── outputs/
    ├── models/
    │   └── mlr_model.pkl              # Serialized trained model
    └── reports/
        └── metrics.txt                # Evaluation metrics
```

---

## 3. Pipeline Overview

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   STAGE 1    │────▶│     STAGE 2      │────▶│    STAGE 3      │────▶│     STAGE 4      │
│ Data Display │     │  Data Preparation │     │  Model Training │     │ Model Evaluation │
│ (READ-ONLY)  │     │   (Consolidated)  │     │                 │     │                  │
└──────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────┘
  src/data_display      src/data_prep           src/model_train        src/model_eval
  notebooks/01_eda
```

### Stage Flow Diagram (data flow)

```
data/raw/50_startups.csv
        │
        ▼
┌───────────────────┐
│   data_display.py  │  ◀── READ-ONLY: loads, prints info, does NOT modify
│   (Stage 1)        │
└───────────────────┘
        │ (same raw data, untouched)
        ▼
┌───────────────────┐
│   data_prep.py     │  ◀── ALL transformations happen HERE:
│   (Stage 2)        │      ① One-Hot Encode 'State'
│                   │      ② Drop first dummy column (avoid trap)
│                   │      ③ Feature Scaling (StandardScaler)
│                   │      ④ Train-Test Split (80/20, random_state=0)
│                   │      ⑤ Save X_train, X_test, y_train, y_test to data/processed/
└───────────────────┘
        │
        ▼
 data/processed/
   X_train.csv, X_test.csv, y_train.csv, y_test.csv
        │
        ▼
┌───────────────────┐
│  model_train.py    │  ◀── Train Multiple Linear Regression on training set
│  (Stage 3)         │      Save model to outputs/models/mlr_model.pkl
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  model_eval.py     │  ◀── Load trained model, predict on test set,
│  (Stage 4)         │      compute R², MAE, RMSE, write to outputs/reports/
└───────────────────┘
```

---

## 4. Stage Specification

### 4.1 Stage 1 — Data Display & EDA (`src/data_display.py`)

| Property         | Constraint                                                                 |
|------------------|----------------------------------------------------------------------------|
| **Write access** | **FORBIDDEN** — this module must NEVER mutate the raw dataset or write to disk |
| **Allowed ops**  | `pd.read_csv()`, `.head()`, `.info()`, `.describe()`, `.shape`, `.dtypes`, `.isnull().sum()` |
| **Forbidden ops**| `.drop()`, `.fillna()`, `.replace()`, `.map()`, `.apply()`, any in-place modification, any `to_csv()` |
| **Output**       | STDOUT only (console/log) — structured summaries of raw data               |
| **Purpose**      | Give the user a snapshot of the dataset before any preprocessing occurs    |

**Responsibilities:**
- Load `data/raw/50_startups.csv`
- Print: shape, column names, dtypes, first 5 rows, statistical summary, null counts
- Optionally generate lightweight summary statistics (mean, median, std per numeric column)
- Confirm that the data is correctly loaded and complete

---

### 4.2 Stage 2 — Consolidated Data Preparation (`src/data_prep.py`)

| Property         | Constraint                                                                              |
|------------------|-----------------------------------------------------------------------------------------|
| **All transforms** | **MUST be confined to this single module** — no preprocessing logic anywhere else   |
| **Input**        | `data/raw/50_startups.csv` (read-only source)                                           |
| **Output**       | `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`                 |
| **Target variable** | `Profit`                                                                             |

**Responsibilities (in strict order):**

1. **Load raw data** from `data/raw/50_startups.csv` into a *copy* (never mutate the raw file).
2. **One-Hot Encoding on `State`** — convert categorical `State` column into binary indicator columns (e.g., `State_New York`, `State_California`, `State_Florida`).
3. **Handle Dummy Variable Trap** — drop exactly **one** dummy column (e.g., `State_Florida`) to avoid perfect multicollinearity. The resulting k-1 dummy columns plus the original numeric features form the full feature matrix.
4. **Separate features (X) and target (y)** — `y = df['Profit']`; `X` = all other columns.
5. **Feature Scaling** — apply `StandardScaler` (fit on training data only; transform test data with the same scaler) to ensure all features contribute proportionally to the model.
6. **Train-Test Split** — split `X` and `y` into train (80%) and test (20%) sets with `random_state=0` for reproducibility.
7. **Persist processed splits** — save `X_train`, `X_test`, `y_train`, `y_test` as CSV files to `data/processed/`.

> **Design Principle:** No other file in the project performs encoding, scaling, splitting, or dummy-trap handling. If preprocessing needs to be updated, only `data_prep.py` changes.

---

### 4.3 Stage 3 — Model Training (`src/model_train.py`)

| Property         | Value                                                          |
|------------------|----------------------------------------------------------------|
| **Input**        | `data/processed/X_train.csv`, `data/processed/y_train.csv`     |
| **Output**       | `outputs/models/mlr_model.pkl`                                 |
| **Algorithm**    | `sklearn.linear_model.LinearRegression`                        |

**Responsibilities:**
- Load `X_train` and `y_train` from `data/processed/`
- Instantiate `LinearRegression`
- Fit the model on training data
- Serialize the trained model to `outputs/models/mlr_model.pkl` using `joblib.dump()`

---

### 4.4 Stage 4 — Model Evaluation (`src/model_eval.py`)

| Property         | Value                                                          |
|------------------|----------------------------------------------------------------|
| **Input**        | `outputs/models/mlr_model.pkl`, `data/processed/X_test.csv`, `data/processed/y_test.csv` |
| **Output**       | `outputs/reports/metrics.txt`                                  |
| **Metrics**      | R² Score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) |

**Responsibilities:**
- Load the serialized model from `outputs/models/mlr_model.pkl`
- Load `X_test` and `y_test` from `data/processed/`
- Predict on the test set
- Compute and print: R², MAE, RMSE
- Persist metrics to `outputs/reports/metrics.txt`
- (Optional) Print the regression coefficients with feature names for interpretability

---

## 5. Orchestration (`main.py`)

`main.py` is the single entry point that executes all four stages sequentially:

```python
# Pseudocode for main.py
import src.data_display   as stage1
import src.data_prep      as stage2
import src.model_train    as stage3
import src.model_eval     as stage4

def run():
    stage1.run()   # READ-ONLY display
    stage2.run()   # Consolidated data preparation
    stage3.run()   # Model training
    stage4.run()   # Evaluation & reporting
```

---

## 6. Enforcement Checklist

| Rule                                          | Enforced By                                     |
|-----------------------------------------------|-------------------------------------------------|
| Raw data never mutated                        | `data_display.py` has no write operations       |
| All preprocessing in one place                | Only `data_prep.py` touches `data/raw/`         |
| Dummy variable trap handled                   | `data_prep.py` drops one dummy column           |
| Reproducible split                            | `random_state=0` in `train_test_split`          |
| Standardized inputs to model                  | `StandardScaler` applied in `data_prep.py`      |
| Separation of concerns                        | Each `src/` module has a single responsibility  |
| Single source of truth                        | `design.md` is referenced before any code change |
