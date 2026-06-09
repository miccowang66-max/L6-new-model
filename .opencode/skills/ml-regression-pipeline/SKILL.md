---
name: ml-regression-pipeline
description: End-to-end Multiple Linear Regression pipeline with 5 feature selection methods, outlier handling, residual diagnostics, sequential feature analysis, and GitHub Pages dashboard deployment. Use when building regression models, comparing feature selection approaches, analyzing datasets like 50_Startups, or creating ML white papers. Make sure to use this skill whenever the user mentions regression analysis, feature selection, backward elimination, forward selection, RFE, Lasso, mutual information, model diagnostics, or wants to compare multiple ML methods.
license: MIT
compatibility: opencode
metadata:
  workflow: ml-pipeline
  audience: data-scientists
  deliverables: scripts, figures, reports, dashboard, white-paper
---

## What I Do

Guide the agent through a complete ML regression workflow from raw data to production dashboard:

1. **Architecture Design** — `design.md` as single source of truth
2. **Correlation Analysis** — heatmap, scatter, boxplot visualizations
3. **Data Preprocessing** — One-Hot Encoding, dummy variable trap, StandardScaler
4. **5 Feature Selection Methods** — Backward Elimination, Forward Selection, RFE, Lasso L1, Mutual Info
5. **Feature Voting & Importance Ranking** — consensus-based feature ordering
6. **Sequential Feature Addition** — RMSE/R² tracking with optimal count detection
7. **Model Evaluation** — R², Adj. R², MAE, RMSE, residual diagnostics
8. **Model Refinement** — Cook's D outlier removal, Box-Cox, Huber regression
9. **Diagnostics Suite** — VIF, normality, Durbin-Watson, cross-validation, state-level slopes
10. **GitHub Pages Dashboard** — interactive HTML with all figures embedded
11. **Technical White Paper** — comprehensive `WHITEPAPER.md`
12. **README with Badges** — matching L6 project format

## Directory Structure

```
project-root/
├── design.md                    # Architecture & pipeline spec
├── WHITEPAPER.md                # Technical white paper
├── README.md                    # GitHub README with badges + live demo
├── requirements.txt
├── index.html                   # GitHub Pages interactive dashboard
├── main_analysis.py             # Full ML pipeline (correlation → eval)
├── feature_selection.py         # 5-method feature selection comparison
├── refined_models.py            # Outlier removal + Box-Cox + Huber
├── outcome_visualization.py     # Sequential feature addition charts
├── method_comparison_charts.py  # 5-method comparison dashboard
├── supplement_analysis.py       # VIF, CV, Cook's D diagnostics
├── data/
│   ├── raw/                     # READ-ONLY — original CSV
│   └── processed/               # X_train, X_test, y_train, y_test
├── .github/workflows/
│   └── deploy.yml               # GitHub Pages CI/CD
├── outputs/
│   ├── figures/                 # 11 PNG charts
│   └── reports/                 # 6 text/CSV reports
└── .opencode/skills/
    └── ml-regression-pipeline/  # This skill
```

---

## Stage 1 — Data Display (READ-ONLY)

- Load raw CSV with `pd.read_csv()`
- Print: shape, dtypes, head(5), describe(), isnull().sum(), value_counts for categorical
- Generate correlation heatmap (seaborn), scatter plots per numeric feature, boxplot per category
- **NEVER mutate raw data** — no `.drop()`, `.fillna()`, `.replace()`, `.apply()`, in-place ops, `.to_csv()`

## Stage 2 — Consolidated Data Preparation

ALL preprocessing MUST be in ONE module. No scatter.

1. `df = df_raw.copy()` — never mutate raw
2. One-Hot Encode categorical columns via `pd.get_dummies(..., dtype=int)`
3. Drop exactly ONE dummy column (k-1 rule, avoid dummy variable trap)
4. Separate X (features) and y (target)
5. `StandardScaler` on numeric features only (NOT dummy columns — they are already 0/1)
6. `train_test_split(test_size=0.2, random_state=0)` for reproducibility
7. Save to `data/processed/`

## Stage 3 — 5 Feature Selection Methods (CORE)

### Method 1: Backward Elimination (P-value, α=0.05)

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

Starts with ALL features, removes the highest P-value feature each iteration, stops when all P ≤ α.

### Method 2: Forward Selection (P-value, α=0.05)

```python
def forward_selection(X, y, sl=0.05):
    remaining = set(X.columns)
    selected = []
    while remaining:
        best_pval, best_feat = float('inf'), None
        for feat in remaining:
            candidates = selected + [feat]
            model = sm.OLS(y, sm.add_constant(X[candidates].astype(float))).fit()
            feat_pval = model.pvalues[feat]
            if feat_pval < best_pval:
                best_pval, best_feat = feat_pval, feat
        if best_pval < sl and best_feat is not None:
            selected.append(best_feat); remaining.remove(best_feat)
        else:
            break
    return selected
```

Starts EMPTY, adds the lowest P-value feature each iteration, stops when no remaining feature has P < α.

### Method 3: Recursive Feature Elimination (RFE)

```python
cv = KFold(n_splits=5, shuffle=True, random_state=42)
best_n, best_score = 1, -999
for n_feat in range(1, X_train.shape[1] + 1):
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=n_feat)
    rfe.fit(X_train, y_train)
    scores = cross_val_score(LinearRegression(), X_train.iloc[:, rfe.support_], y_train, cv=cv, scoring='r2')
    if scores.mean() > best_score:
        best_score, best_n = scores.mean(), n_feat

rfe = RFE(estimator=LinearRegression(), n_features_to_select=best_n).fit(X_train, y_train)
sel_rfe = [f for f, s in zip(feature_names, rfe.support_) if s]
```

Grid search over n_features_to_select, picking the count that maximizes 5-fold CV R².

### Method 4: Lasso Regression (L1 Regularization)

```python
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-4, 2, 50))
lasso_cv.fit(X_train, y_train)
sel_lasso = [f for f, c in zip(feature_names, lasso_cv.coef_) if abs(c) > 1e-5]
```

L1 penalty shrinks unimportant coefficients to exactly zero. Alpha tuned via 5-fold CV.

### Method 5: Mutual Information (non-linear dependency)

```python
mi = mutual_info_regression(X_train, y_train, random_state=42)
mi_threshold = mi.max() * 0.20
sel_mi = [f for f, m in zip(feature_names, mi) if m >= mi_threshold]
```

Captures non-linear relationships that Pearson r misses. Threshold = 20% of max MI.

## Stage 4 — Feature Importance Voting

```python
methods = {
    "Backward Elimination": sel_back,
    "Forward Selection": sel_forward,
    "RFE (CV-optimal)": sel_rfe,
    "Lasso L1": sel_lasso,
    "Mutual Info (top)": sel_mi,
}

feature_votes = {}
for method_name, selected in methods.items():
    for feat in selected:
        feature_votes[feat] = feature_votes.get(feat, 0) + 1

feature_importance = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
```

**CRITICAL**: Order features by votes descending. R&D Spend-style features (5/5) go first, noise features (1/5) go last.

## Stage 5 — Sequential Feature Addition Analysis

```python
feature_order = [f for f, v in feature_importance]

for i in range(1, len(feature_order) + 1):
    selected = feature_order[:i]
    lr = LinearRegression().fit(X_train[selected], y_train)
    y_pred = lr.predict(X_test[selected])
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

This reveals the **optimal feature count** — where RMSE bottoms out and R² peaks. Adding more features beyond this point degrades performance (overfitting).

## Stage 6 — Model Refinement

### 6a: Outlier Removal (Cook's Distance)

```python
model_full = sm.OLS(y_all, sm.add_constant(X_all_scaled)).fit()
influence = model_full.get_influence()
cooks_d = influence.cooks_distance[0]
high_influence_idx = np.where(cooks_d > 4/len(df))[0]

# Rerun entire pipeline without these rows
df_clean = df_raw.drop(index=high_influence_idx).reset_index(drop=True)
```

Cook's D > 4/n flags high-leverage points. Removing them often reveals hidden significant features (e.g., Marketing Spend P-value went from 0.07 → 0.04).

### 6b: Box-Cox Transformation

```python
y_bc, lambda_bc = stats.boxcox(y_all)
```

Only needed when Omnibus normality p < 0.05. Transform y, rerun backward elimination, inverse-transform predictions for metrics.

### 6c: Huber Robust Regression

```python
huber = HuberRegressor(epsilon=1.35, max_iter=500)
huber.fit(X_train[selected], y_train)
```

Resistant to outliers without removing data points. Best for production when future outliers are expected.

## Stage 7 — Evaluation & Visualizations

### Required Charts (all saved to `outputs/figures/`)

| # | Filename | Content |
|---|----------|---------|
| 1 | `corr_heatmap.png` | Pearson correlation heatmap (masked upper triangle) |
| 2 | `scatter_features.png` | All numeric features vs target scatter plots with r values |
| 3 | `boxplot_state.png` | Categorical variable vs target boxplots |
| 4 | `method_comparison_rmse_r2.png` | Horizontal bar charts: RMSE & R² for all 5 methods |
| 5 | `method_comparison_heatmap_performance.png` | Feature selection heatmap + features vs performance dual-axis |
| 6 | `method_comparison_dashboard.png` | 4-in-1: RMSE, R², heatmap, summary table |
| 7 | `feature_selection_comparison.png` | Feature votes heatmap + Test R² bar chart |
| 8 | `feature_count_performance.png` | Dual line chart: RMSE/R² vs Number of Features |
| 9 | `pred_vs_actual.png` | Scatter plot with y=x reference line |
| 10 | `residuals.png` | Residuals vs predicted + residuals histogram |
| 11 | `model_comparison.png` | Refined model comparison (A/B/C/D) |

### Required Reports

| # | Filename | Content |
|---|----------|---------|
| 1 | `metrics.txt` | Full OLS summary + evaluation metrics |
| 2 | `refined_models.txt` | All refined model results + comparison table |
| 3 | `feature_selection.txt` | 5-method votes + feature importance ranking |
| 4 | `feature_selection_results.csv` | Sequential addition data (Excel compatible) |
| 5 | `feature_selection_results.tsv` | Sequential addition data (TSV format) |
| 6 | `method_comparison_results.csv` | Full 5-method comparison table |

### Evaluation Metrics Table

Sort by R² descending, highlight best row:

| Method | n_Feat | Selected Features | RMSE | R² | Adj.R² | CV R² |
|--------|--------|-------------------|------|-----|--------|-------|

## Stage 8 — Diagnostics Suite

| Diagnostic | Tool | Threshold | Action if Failed |
|------------|------|-----------|------------------|
| **Cook's Distance** | `statsmodels` influence | > 4/n | Remove rows, rerun pipeline |
| **Residual Normality** | Omnibus test | p < 0.05 | Box-Cox transform y |
| **Autocorrelation** | Durbin-Watson | < 1.5 or > 2.5 | Check data ordering, use HAC standard errors |
| **Multicollinearity** | VIF | > 10 | Remove or combine collinear features |
| **Overfitting** | 5-fold CV R² vs Test R² | Gap > 0.05 | Reduce feature count, use regularization |
| **State-level slopes** | Per-group OLS | Slope variance across groups | If large variance → keep interaction term |

## Stage 9 — Deployment

### GitHub Pages Dashboard

Create `index.html` using Tailwind CSS CDN with:
- Dark theme (obsidian/charcoal), green accent (#008060)
- Inter + Roboto Mono fonts via Google Fonts
- Hero section with 4 key metric cards (Best R², Best RMSE, Methods, Optimal Features)
- Sections embedding all 11 figures with descriptive captions
- Insight callout boxes for each section
- "View on GitHub" link

**Hardcoded HTML data tables** (do NOT use images for data — render as HTML `<table>`):

1. **Method Comparison Results table** — all 5 methods with Rank, Method, n_Feat, Selected Features, RMSE, Test R², Adj. R², CV R² Mean, CV R² Std. Green highlight on best row. Red on worst row.
2. **Sequential Feature Addition Results table** — n_Feat 1-5 with Selected Features, RMSE, R². Green highlight on optimal row (2 features).
3. **Feature Votes cards** — 5 cards in a grid with feature name, vote count (X/5), and a progress bar (green for 5/5, yellow for 3/5, gray for 2/5, red for 1/5).

Workflow file `.github/workflows/deploy.yml`:
```yaml
name: Deploy to GitHub Pages
on:
  push:
    branches: [master]
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: "pages"
  cancel-in-progress: false
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v4
      - uses: actions/upload-pages-artifact@v3
        with:
          path: '.'
      - id: deployment
        uses: actions/deploy-pages@v4
```

### README Format

Follow L3-Personal-Web-Page style:
- Title with emoji + badges (License, Python, scikit-learn, GitHub Pages)
- Blockquote description
- Live Demo table (GitHub Pages + GitHub Repo)
- Features section with categorized subsections
- Key Findings table (feature votes)
- Quick Start code block
- Project Structure tree
- Tech Stack table
- Results Summary tables
- Customization table
- MIT License footer

### White Paper

Generate `WHITEPAPER.md` sections:
1. Project Overview (background, objectives, core question, tech stack)
2. System Architecture (pipeline diagram, directory tree, design principles)
3. Data Specification (column schema, quality report, statistical summary)
4. Data Preprocessing (step-by-step pipeline, encoding spec, scaling strategy)
5. Feature Selection Methodology (5 method details, voting mechanism)
6. Model Implementation (algorithm, variants, evaluation metrics)
7. Experimental Results (all data tables, final equation, feature votes)
8. Model Diagnostics (baseline vs cleaned diagnostics, CV stability, state slopes)
9. Deployment Architecture (GitHub Pages CI/CD, Agent Skill)
10. Appendix (figures list, reports list, glossary)

## Key Insights to Report

1. **Unanimous features** (5/5 votes) — iron-clad predictors
2. **Consensus features** (≥3/5) — strong evidence, worth including
3. **Marginal features** (≤2/5) — weak evidence, probably noise
4. **Optimal feature count** — where RMSE is lowest AND R² is highest
5. **Overfitting boundary** — point after which adding features hurts performance
6. **Outlier impact** — how many removed, what changed after removal
7. **Production recommendation** — which model variant to use

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

- Building a Multiple Linear Regression model from a CSV dataset with categorical + numeric features
- Need to compare 5 different feature selection methods and vote on importance
- Want sequential feature addition analysis to find optimal feature count
- Need comprehensive diagnostics (normality, outliers, multicollinearity, autocorrelation)
- Want to deploy an interactive GitHub Pages dashboard with all figures
- Need to generate a technical white paper (`WHITEPAPER.md`)
- User asks for "feature selection", "backward elimination", "which features matter", "regression", "R²", "RMSE"
- User wants to understand feature importance ranking or compare ML methods
- User mentions "50_Startups", "startup profit prediction", or similar regression datasets

## Example Workflow

```
1.  Create design.md → directory structure
2.  Load data → One-Hot Encode → StandardScaler → Train/Test Split
3.  Correlation analysis → heatmap + scatter + boxplot
4.  Run all 5 feature selection methods
5.  Create feature vote count + importance ranking
6.  Sequential addition analysis (1→2→...→all features)
7.  Generate 11 charts to outputs/figures/
8.  Generate 6 reports to outputs/reports/
9.  Run diagnostics (Cook's D, normality, VIF, CV, DW, state slopes)
10. Run refined models (outlier removal, Box-Cox, Huber, No_RD flag)
11. Create index.html dashboard
12. Create .github/workflows/deploy.yml
13. Write WHITEPAPER.md (10 sections)
14. Write README.md (matching L3 format with badges)
15. Create .gitignore (exclude .env)
16. Push to GitHub, enable GitHub Pages → live at username.github.io/repo-name/
```
