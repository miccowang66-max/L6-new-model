---
name: ml-regression-pipeline
description: End-to-end Multiple Linear Regression pipeline with 5 feature selection methods, outlier handling, residual diagnostics, and sequential feature analysis. Use when building regression models, comparing feature selection approaches, or analyzing datasets like 50_Startups. Make sure to use this skill whenever the user mentions regression analysis, feature selection, backward elimination, or wants to compare multiple ML methods.
license: MIT
compatibility: opencode
metadata:
  workflow: ml-pipeline
  audience: data-scientists
---

## What I Do

Guide the agent through a complete ML regression workflow with comprehensive feature selection comparison:

1. Correlation analysis with visualizations (heatmap, scatter, boxplot)
2. Data preprocessing (One-Hot Encoding, dummy variable trap avoidance, StandardScaler)
3. **5 Feature Selection Methods** comparison (Backward Elimination, Forward Selection, RFE, Lasso, Mutual Information)
4. Sequential feature addition analysis (RMSE/R² by feature count)
5. Model evaluation (R², Adj. R², MAE, RMSE, residual diagnostics)
6. Optional refinements: outlier removal, Box-Cox transform, Huber regression

## Directory Structure

Always enforce this layout before writing any code:

```
project-root/
├── design.md                # Single source of truth (pipeline + architecture)
├── requirements.txt
── main.py                  # Orchestrator
├── data/
│   ├── raw/                 # READ-ONLY — original CSV
│   └── processed/           # X_train, X_test, y_train, y_test
├── src/
│   ├── data_display.py      # Stage 1: READ-ONLY exploration
│   ├── data_prep.py         # Stage 2: ALL transformations HERE
│   ├── model_train.py       # Stage 3: Training
│   ── model_eval.py        # Stage 4: Evaluation
├── notebooks/
│   ── 01_eda.ipynb
── outputs/
    ├── figures/
    ├── models/
    └── reports/
```

## Pipeline Rules

### Stage 1 — Data Display (READ-ONLY)
- Load raw CSV with `pd.read_csv()`
- Print: shape, dtypes, head(5), describe(), isnull().sum()
- Generate correlation heatmap (seaborn), scatter plots, boxplots
- **NEVER** mutate raw data — no `.drop()`, `.fillna()`, `.replace()`, `.apply()`, in-place ops, or `.to_csv()`

### Stage 2 — Consolidated Data Preparation
All preprocessing MUST live in a single `data_prep.py`. Do NOT scatter across files.

1. Copy raw dataframe (`df = df_raw.copy()`)
2. One-Hot Encode categorical columns via `pd.get_dummies(..., dtype=int)`
3. Drop exactly ONE dummy column to avoid the dummy variable trap (k-1 rule)
4. Separate X (features) and y (target)
5. `StandardScaler` on numeric features only (not dummy columns)
6. `train_test_split(test_size=0.2, random_state=0)`
7. Save `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` to `data/processed/`

### Stage 3 — 5 Feature Selection Methods (CORE)

Run ALL five methods and compare results. This is the heart of the analysis.

#### Method 1: Backward Elimination (P-value)
```python
def backward_elimination(X, y, sl=0.05):
    X_be = sm.add_constant(X.copy())
    features = list(X.columns)
    while True:
        model = sm.OLS(y, X_be.astype(float)).fit()
        pvalues = model.pvalues.drop("const", errors="ignore")
        max_pval = pvalues.max()
        if max_pval > sl:
            max_feat = pvalues.idxmax()
            X_be = X_be.drop(columns=[max_feat])
            features.remove(max_feat)
        else:
            break
    return [f for f in X.columns if f in features]
```

#### Method 2: Forward Selection (P-value)
```python
def forward_selection(X, y, sl=0.05):
    remaining = set(X.columns)
    selected = []
    while remaining:
        best_pval = float('inf')
        best_feat = None
        for feat in remaining:
            candidates = selected + [feat]
            X_sub = sm.add_constant(X[candidates].astype(float))
            model = sm.OLS(y, X_sub).fit()
            pvals = model.pvalues.drop("const", errors="ignore")
            if pvals[feat] < best_pval:
                best_pval = pvals[feat]
                best_feat = feat
        if best_pval < sl and best_feat is not None:
            selected.append(best_feat)
            remaining.remove(best_feat)
        else:
            break
    return selected
```

#### Method 3: Recursive Feature Elimination (RFE)
```python
# Find optimal n_features via CV
best_n = 1
best_score = -999
for n_feat in range(1, X_train.shape[1] + 1):
    lr = LinearRegression()
    rfe = RFE(estimator=lr, n_features_to_select=n_feat)
    rfe.fit(X_train, y_train)
    scores = cross_val_score(lr, X_train.iloc[:, rfe.support_], y_train, cv=5, scoring='r2')
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_n = n_feat

# Final RFE with best n
rfe = RFE(estimator=LinearRegression(), n_features_to_select=best_n)
rfe.fit(X_train, y_train)
sel_rfe = [f for f, s in zip(feature_names, rfe.support_) if s]
```

#### Method 4: Lasso Regression (L1)
```python
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-4, 2, 50))
lasso_cv.fit(X_train, y_train)
sel_lasso = [f for f, c in zip(feature_names, lasso_cv.coef_) if abs(c) > 1e-5]
```

#### Method 5: Mutual Information
```python
mi = mutual_info_regression(X_train, y_train, random_state=42)
mi_threshold = mi.max() * 0.20  # 20% of max MI
sel_mi = [f for f, m in zip(feature_names, mi) if m >= mi_threshold]
```

### Stage 4 — Feature Importance Voting & Ordering

After running all 5 methods, create a **feature vote count**:

```python
# Count how many methods selected each feature
feature_votes = {}
for method_name, selected in methods.items():
    for feat in selected:
        feature_votes[feat] = feature_votes.get(feat, 0) + 1

# Sort by votes (descending)
feature_importance = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
```

**CRITICAL**: Always order features by importance (votes) when presenting results. Most important first.

### Stage 5 — Sequential Feature Addition Analysis

Add features one-by-one in importance order and track performance:

```python
feature_order = [f for f, v in feature_importance]  # Most important first

results = []
for i in range(1, len(feature_order) + 1):
    selected = feature_order[:i]
    lr = LinearRegression().fit(X_train[selected], y_train)
    y_pred = lr.predict(X_test[selected])
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append({
        "Number of Features": i,
        "Selected Features": selected,
        "RMSE": rmse,
        "R-squared": r2
    })
```

### Stage 6 — Evaluation & Visualization

#### Required Visualizations

**1. Method Comparison (RMSE & R² bars)**
- Horizontal bar charts comparing all 5 methods
- RMSE on left, R² on right
- Color-coded by performance

**2. Feature Selection Heatmap**
- Rows: 5 methods
- Columns: All features
- Cells: 1 (selected) or 0 (not selected)
- Use RdYlGn colormap

**3. Sequential Addition Performance**
- Left: RMSE vs Number of Features (line chart)
- Right: R² vs Number of Features (line chart)
- Show value labels on each point
- Identify optimal feature count (lowest RMSE / highest R²)

**4. Combined Dashboard (4-in-1)**
- Top-left: RMSE bar chart
- Top-right: R² bar chart
- Bottom-left: Feature selection heatmap
- Bottom-right: Summary table with all metrics

#### Required Metrics Table

| Method | n_Feat | Selected Features | RMSE | R² | Adj.R² | CV R² |
|--------|--------|-------------------|------|-----|--------|-------|

Sort by R² descending. Highlight best row.

### Stage 7 — Critical Diagnostics

Always run these after initial model:

1. **Cook's Distance** — identify high-leverage points (`> 4/n`). Rerun without them.
2. **Residual Normality** — Omnibus test p-value; if p < 0.05, try Box-Cox on target.
3. **Durbin-Watson** — should be near 2.0; if < 1.5, check for autocorrelation.
4. **VIF** — ensure no feature has VIF > 10 (multicollinearity threshold).
5. **Cross-validation** — use 5-fold CV to verify R² stability.

## Key Insights to Report

Always include these findings in the final report:

1. **Unanimous features** (selected by all 5 methods) — these are iron-clad predictors
2. **Consensus features** (selected by ≥3 methods) — strong evidence
3. **Marginal features** (selected by 1-2 methods) — weak evidence, may be noise
4. **Optimal feature count** — where RMSE is lowest and R² is highest
5. **Overfitting warning** — if adding features degrades performance

## Dependencies

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
statsmodels>=0.14
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
```

## When to Use Me

- Building a Multiple Linear Regression model from a CSV dataset
- Need to compare multiple feature selection methods
- Want to find optimal feature count via sequential addition
- Need comprehensive diagnostics (normality, outliers, multicollinearity)
- Working with datasets containing categorical variables and numeric features
- User asks for "feature selection", "backward elimination", "which features matter"
- User wants to understand feature importance ranking

## Example Workflow

```
1. Load data → One-Hot Encode → StandardScaler → Train/Test Split
2. Run all 5 feature selection methods
3. Create feature vote count and importance ranking
4. Sequential addition analysis (1 feature → 2 features → ... → all)
5. Generate 4 visualizations (method comparison, heatmap, sequential, dashboard)
6. Run diagnostics (Cook's D, normality, VIF, CV)
7. Report: unanimous features, optimal count, overfitting warnings
```
