# L6 Crisp-RD2 — ML Regression Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://github.com/miccowang66-max/L6-new-model/blob/master/LICENSE) [![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://www.python.org/downloads/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square)](https://scikit-learn.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io/) [![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-7c3aed?style=flat-square&logo=github)](https://miccowang66-max.github.io/L6-new-model/)

> A production-ready Multiple Linear Regression pipeline with 5 feature selection methods, outlier handling, residual diagnostics, comprehensive visualizations, and an interactive Streamlit dashboard.
> **Python · scikit-learn · statsmodels · seaborn · Plotly · Streamlit · One-Hot Encoding · Backward Elimination**

---

## 🌐 Live Demos

| Platform | URL | Description |
|----------|-----|-------------|
| **Streamlit App** | `streamlit run app.py` | Interactive CRISP-DM dashboard with data exploration, modeling workflow, and SFA charts |
| **GitHub Pages** | [miccowang66-max.github.io/L6-new-model](https://miccowang66-max.github.io/L6-new-model/) | Interactive dashboard with all figures and analysis |
| **GitHub Repo** | [miccowang66-max/L6-new-model](https://github.com/miccowang66-max/L6-new-model) | Full source code, scripts, and reports |

---

## ✨ Features

### Analysis Pipeline

- **Correlation Analysis** — Pearson heatmap, scatter plots, boxplots for EDA
- **Consolidated Preprocessing** — One-Hot Encoding, dummy variable trap avoidance, StandardScaler
- 📊 **5 Feature Selection Methods** — Backward Elimination, Forward Selection, RFE, Lasso L1, Mutual Information
- 📈 **Sequential Feature Addition** — RMSE/R² tracking by feature count with optimal point detection
- **Model Diagnostics** — Cook's Distance, residual normality (Omnibus), Durbin-Watson, VIF, 5-fold CV

### Visualizations

- 📉 **Method Comparison** — Horizontal bar charts for RMSE and R² across all 5 methods
- 🔥 **Feature Selection Heatmap** — Vote matrix showing which methods selected which features
- **Sequential Performance** — Line charts tracking RMSE/R² as features are added
- 🎯 **Combined Dashboard** — 4-in-one overview (RMSE, R², heatmap, summary table)
- 📋 **Excel-Ready Tables** — CSV/TSV exports for all results

### Streamlit App (app.py)

- 📊 **CRISP-DM Dashboard** — 4-chapter interactive web app
- 📋 **Data Discovery** — Raw data explorer with statistical summaries and KPI cards
- ⚙️ **Modeling Workflow** — Step-by-step pipeline visualization (Cleaning → Encoding → Split → Modeling)
- 🔬 **Feature Selection Analysis** — Interactive SFA table with dual Plotly line charts (RMSE & R²)
- 🏆 **Optimal Model Detection** — Auto-highlighted elbow point with insight block

### Key Findings

| Feature | Votes (out of 5) | Status |
|---------|:---:|--------|
| R&D Spend | **5/5** | Unanimous — iron-clad predictor |
| Marketing Spend | 3/5 | Consensus — marginal utility |
| State_Florida | 2/5 | Weak evidence |
| Administration | 1/5 | Noise |
| State_New York | 1/5 | Noise |

**Optimal Model**: 2 features [R&D Spend, Marketing Spend]
**Test R²**: 0.9474 | **RMSE**: $8,198.80

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/miccowang66-max/L6-new-model.git
cd L6-new-model

# Install dependencies
pip install -r requirements.txt

# Run full analysis pipeline
python main_analysis.py

# Run 5-method feature selection comparison
python feature_selection.py

# Run refined models (outlier removal + Box-Cox + Huber)
python refined_models.py

# Generate outcome charts
python outcome_visualization.py

# Generate method comparison charts
python method_comparison_charts.py

# Launch interactive Streamlit dashboard
streamlit run app.py
```

---

## 📁 Project Structure

```
L6-new-model/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── app.py                              # Streamlit interactive dashboard
├── main_analysis.py                    # Full ML pipeline (correlation → eval)
├── feature_selection.py                # 5-method feature selection comparison
├── refined_models.py                   # Outlier removal + Box-Cox + Huber
├── outcome_visualization.py            # Sequential feature addition charts
├── method_comparison_charts.py         # 5-method comparison dashboard
├── supplement_analysis.py              # VIF, CV, Cook's D diagnostics
├── data/
│   ├── raw/
│   │   └── 50_startups.csv            # READ-ONLY source data
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── outputs/
│   ├── figures/                        # 11 PNG charts
│   │   ├── corr_heatmap.png
│   │   ├── scatter_features.png
│   │   ├── boxplot_state.png
│   │   ├── pred_vs_actual.png
│   │   ├── residuals.png
│   │   ├── feature_count_performance.png
│   │   ├── feature_selection_comparison.png
│   │   ├── method_comparison_rmse_r2.png
│   │   ├── method_comparison_heatmap_performance.png
│   │   ├── method_comparison_dashboard.png
│   │   └── model_comparison.png
│   └── reports/                        # 6 text/CSV reports
│       ├── metrics.txt
│       ├── refined_models.txt
│       ├── feature_selection.txt
│       ├── feature_selection_results.csv
│       ├── feature_selection_results.tsv
│       └── method_comparison_results.csv
└── .opencode/skills/                   # Reusable ML skills
    ├── ml-regression-pipeline/
    │   └── SKILL.md
    └── skill-creator/
        ├── SKILL.md
        ├── agents/
        ├── scripts/
        ├── eval-viewer/
        └── assets/
```

---

## 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| [Python 3.10+](https://www.python.org/) | Core language |
| [pandas](https://pandas.pydata.org/) | Data manipulation |
| [numpy](https://numpy.org/) | Numerical computing |
| [scikit-learn](https://scikit-learn.org/) | ML models, preprocessing, CV |
| [statsmodels](https://www.statsmodels.org/) | OLS regression, P-values, diagnostics |
| [matplotlib](https://matplotlib.org/) | Static visualizations |
| [seaborn](https://seaborn.pydata.org/) | Statistical visualizations |
| [scipy](https://scipy.org/) | Box-Cox transform, statistical tests |
| [Streamlit](https://streamlit.io/) | Interactive web dashboard |
| [Plotly](https://plotly.com/) | Interactive charts (hover, zoom, pan) |

---

## 📊 Results Summary

### 5 Feature Selection Methods Comparison

| Method | n_Feat | Selected Features | RMSE | R² | Adj.R² |
|--------|:---:|---|---:|---:|---:|
| **Mutual Info** | 2 | R&D + Marketing | 8,198.80 | **0.9474** | 0.9324 |
| Backward Elim. | 1 | R&D | 8,274.87 | 0.9465 | 0.9398 |
| Forward Select. | 1 | R&D | 8,274.87 | 0.9465 | 0.9398 |
| RFE | 3 | R&D + Marketing + FL | 8,376.45 | 0.9451 | 0.9177 |
| Lasso L1 | 5 | All features | 9,137.99 | 0.9347 | 0.8531 |

### Sequential Feature Addition

| Features | RMSE | R² |
|:---:|---:|---:|
| 1 (R&D) | 8,274.87 | 0.9465 |
| **2 (R&D + Marketing)** | **8,198.80** | **0.9474** |
| 3 (+ State_FL) | 8,376.45 | 0.9451 |
| 4 (+ Admin) | 9,068.54 | 0.9357 |
| 5 (+ State_NY) | 9,137.99 | 0.9347 |

---

## 🔧 Customization

| Element | How to Change |
|---------|---------------|
| **Dataset** | Replace `data/raw/50_startups.csv` with your CSV |
| **Target variable** | Change `TARGET = "Profit"` in `main_analysis.py` |
| **Significance level** | Change `SIGNIFICANCE_LEVEL = 0.05` in backward elimination |
| **Train/test split** | Change `TEST_SIZE = 0.2` and `RANDOM_STATE = 0` |
| **Feature selection methods** | Add/remove methods in `feature_selection.py` |
| **Visualizations** | Modify matplotlib/seaborn parameters in chart scripts |
| **Report format** | Edit CSV/TSV output in `outcome_visualization.py` |
| **Streamlit dashboard** | Modify `app.py` for custom UI, data, or charts |

---

## 📄 License

MIT © [miccowang66-max](https://github.com/miccowang66-max)
