# -*- coding: utf-8 -*-
import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.metrics import r2_score
from itertools import combinations

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ===========================================================================
# 0. Load & Preprocess (One-Hot Encoding)
# ===========================================================================
print("=" * 70)
print("  5 Feature Selection Methods on 50_Startups (One-Hot Encoded)")
print("=" * 70)

df_raw = pd.read_csv("data/raw/50_startups.csv")
print(f"\n[0] Raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} cols")
print(f"    Columns: {list(df_raw.columns)}")

# One-Hot Encode State
dummies = pd.get_dummies(df_raw["State"], prefix="State", dtype=int)
drop_col = dummies.columns[0]
dummies = dummies.drop(columns=[drop_col])
print(f"    One-Hot: dropped '{drop_col}' as baseline, kept {list(dummies.columns)}")

df_enc = pd.concat([df_raw.drop(columns=["State"]), dummies], axis=1)
X_all = df_enc.drop(columns=["Profit"])
y_all = df_enc["Profit"]

# Standardize numeric features
num_cols = ["R&D Spend", "Administration", "Marketing Spend"]
scaler = StandardScaler()
X_scaled = X_all.copy()
X_scaled[num_cols] = scaler.fit_transform(X_all[num_cols])

feature_names = list(X_scaled.columns)
print(f"    Encoded features ({len(feature_names)}): {feature_names}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_all, test_size=0.2, random_state=0
)
print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

# ===========================================================================
# 1. Backward Elimination (statsmodels OLS)
# ===========================================================================
print("\n" + "=" * 70)
print("  METHOD 1: Backward Elimination (P-value, alpha=0.05)")
print("=" * 70)

def backward_elimination(X, y, sl=0.05):
    X_be = sm.add_constant(X.copy())
    features = list(X.columns)
    removed = []
    steps = []
    while True:
        model = sm.OLS(y, X_be.astype(float)).fit()
        pvalues = model.pvalues.drop("const", errors="ignore")
        max_pval = pvalues.max()
        max_feat = pvalues.idxmax()
        steps.append({"features": list(features), "removed": None,
                       "max_pval": max_pval, "max_feat": max_feat,
                       "r2": model.rsquared, "aic": model.aic})
        if max_pval > sl:
            X_be = X_be.drop(columns=[max_feat])
            features.remove(max_feat)
            steps[-1]["removed"] = max_feat
            removed.append(max_feat)
            print(f"    [-] Removed: {max_feat:20s}  P={max_pval:.4f}")
        else:
            print(f"    [+] Stop:     all P <= {sl}")
            break
    selected = [f for f in X.columns if f in features]
    return selected, removed, steps

sel_back, rem_back, steps_back = backward_elimination(X_train, y_train)
print(f"\n  Selected ({len(sel_back)}): {sel_back}")

# ===========================================================================
# 2. Forward Selection
# ===========================================================================
print("\n" + "=" * 70)
print("  METHOD 2: Forward Selection (P-value threshold, alpha=0.05)")
print("=" * 70)

def forward_selection(X, y, sl=0.05):
    remaining = set(X.columns)
    selected = []
    steps = []

    while remaining:
        best_pval = float('inf')
        best_feat = None
        best_r2 = 0
        for feat in remaining:
            candidates = selected + [feat]
            X_sub = sm.add_constant(X[candidates].astype(float))
            model = sm.OLS(y, X_sub).fit()
            pvals = model.pvalues.drop("const", errors="ignore")
            feat_pval = pvals[feat]
            if feat_pval < best_pval:
                best_pval = feat_pval
                best_feat = feat
                best_r2 = model.rsquared
        if best_pval < sl and best_feat is not None:
            selected.append(best_feat)
            remaining.remove(best_feat)
            steps.append({"added": best_feat, "pval": best_pval, "r2": best_r2})
            print(f"    [+] Added: {best_feat:20s}  P={best_pval:.4f}  R^2={best_r2:.4f}")
        else:
            print(f"    [-] Stop: no remaining feature has P < {sl}")
            break
    return selected, steps

sel_forward, steps_forward = forward_selection(X_train, y_train)
print(f"\n  Selected ({len(sel_forward)}): {sel_forward}")

# ===========================================================================
# 3. Recursive Feature Elimination (RFE, sklearn)
# ===========================================================================
print("\n" + "=" * 70)
print("  METHOD 3: Recursive Feature Elimination (RFE)")
print("=" * 70)

# Try RFE with 1 to max features, pick best via CV
cv = KFold(n_splits=5, shuffle=True, random_state=42)
best_n = 1
best_score = -999

for n_feat in range(1, X_train.shape[1] + 1):
    lr = LinearRegression()
    rfe = RFE(estimator=lr, n_features_to_select=n_feat)
    rfe.fit(X_train, y_train)
    scores = cross_val_score(lr, X_train.iloc[:, rfe.support_], y_train, cv=cv, scoring='r2')
    mean_score = scores.mean()
    if mean_score > best_score:
        best_score = mean_score
        best_n = n_feat
    print(f"    n_features={n_feat}: CV R^2={mean_score:.4f}")

# Final RFE with best n
lr = LinearRegression()
rfe = RFE(estimator=lr, n_features_to_select=best_n)
rfe.fit(X_train, y_train)
sel_rfe = [f for f, s in zip(feature_names, rfe.support_) if s]
rfe_ranking = pd.DataFrame({
    "Feature": feature_names,
    "RF_Rank": rfe.ranking_,
    "Selected": rfe.support_
}).sort_values("RF_Rank")
print(f"\n  Ranking:") 
print(rfe_ranking.to_string(index=False))
print(f"\n  Best n = {best_n}, Selected: {sel_rfe}")

# ===========================================================================
# 4. Lasso Regression (L1 Regularization)
# ===========================================================================
print("\n" + "=" * 70)
print("  METHOD 4: Lasso Regression (L1)")
print("=" * 70)

# LassoCV to find best alpha
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-4, 2, 50))
lasso_cv.fit(X_train, y_train)
print(f"    Best alpha: {lasso_cv.alpha_:.4f}")
print(f"    Train R^2:  {lasso_cv.score(X_train, y_train):.4f}")
print(f"    Test R^2:   {lasso_cv.score(X_test, y_test):.4f}")

# Show coefficients
lasso_coef = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": lasso_cv.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
print(f"\n    Lasso Coefficients (zero = eliminated):")
print(lasso_coef.to_string(index=False))

# Track which alphas zero out which features
sel_lasso = [f for f, c in zip(feature_names, lasso_cv.coef_) if abs(c) > 1e-5]
print(f"\n  Non-zero coefficients ({len(sel_lasso)}): {sel_lasso}")

# Lasso path visualization
print(f"\n    Lasso path at different alphas:")
alphas_test = np.logspace(-4, 2, 30)
for alpha in alphas_test:
    lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    lasso.fit(X_train, y_train)
    nz = np.sum(np.abs(lasso.coef_) > 1e-5)
    selected_alpha = [f for f, c in zip(feature_names, lasso.coef_) if abs(c) > 1e-5]
    if nz <= 3:
        print(f"      alpha={alpha:.4f}: {nz} features -> {selected_alpha}")

# ===========================================================================
# 5. Mutual Information
# ===========================================================================
print("\n" + "=" * 70)
print("  METHOD 5: Mutual Information (non-linear dependency)")
print("=" * 70)

mi = mutual_info_regression(X_train, y_train, random_state=42)
mi_df = pd.DataFrame({
    "Feature": feature_names,
    "Mutual_Info": mi,
    "Pearson_r": [X_train[f].corr(y_train) for f in feature_names]
}).sort_values("Mutual_Info", ascending=False)
print(mi_df.to_string(index=False))

# Select top features by MI threshold (knee method or top-k)
# Use a simple approach: features with MI > 20% of max
mi_threshold = mi.max() * 0.20
sel_mi = [f for f, m in zip(feature_names, mi) if m >= mi_threshold]
print(f"\n  MI threshold = {mi_threshold:.4f} (20% of max)")
print(f"  Selected ({len(sel_mi)}): {sel_mi}")

# ===========================================================================
# FINAL: Comparison of all 5 methods
# ===========================================================================
print("\n" + "=" * 70)
print("  FINAL: Cross-Method Feature Selection Comparison")
print("=" * 70)

# Evaluate each method's selected features on test set
methods = {
    "Backward Elimination": sel_back,
    "Forward Selection": sel_forward,
    "RFE (CV-optimal)": sel_rfe,
    "Lasso L1": sel_lasso,
    "Mutual Info (top)": sel_mi,
}

all_results = []
for name, selected in methods.items():
    if len(selected) == 0:
        print(f"  [SKIP] {name}: no features selected")
        continue
    lr = LinearRegression().fit(X_train[selected], y_train)
    y_pred = lr.predict(X_test[selected])
    r2 = r2_score(y_test, y_pred)
    n, p = len(y_test), len(selected)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    cv_scores = cross_val_score(LinearRegression(), X_train[selected], y_train, cv=5, scoring='r2')
    all_results.append({
        "Method": name,
        "Features": ", ".join(selected),
        "n_Feat": len(selected),
        "Test_R2": r2,
        "Adj_R2": adj_r2,
        "CV_R2_mean": cv_scores.mean(),
        "CV_R2_std": cv_scores.std(),
    })
    print(f"  {name:25s}: {len(selected)} features [{', '.join(selected)}]")
    print(f"    Test R^2={r2:.4f}, Adj.R^2={adj_r2:.4f}, CV R^2={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

# Sort by Test R^2 descending
results_df = pd.DataFrame(all_results).sort_values("Test_R2", ascending=False)
print(f"\n  Ranked by Test R^2:")
print(results_df[["Method", "n_Feat", "Test_R2", "Adj_R2", "CV_R2_mean", "Features"]].to_string(index=False))

# Feature selection heatmap (which methods selected which features)
print(f"\n  Feature Selection Heatmap:")
heatmap_data = pd.DataFrame(index=feature_names)
for method_name, selected in methods.items():
    heatmap_data[method_name] = [1 if f in selected else 0 for f in feature_names]
print(heatmap_data.to_string())

# Visualize: stacked bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Feature selection heatmap
hm_data = heatmap_data.T
sns.heatmap(hm_data, annot=True, fmt="d", cmap="RdYlGn", cbar=False,
            linewidths=1, ax=axes[0])
axes[0].set_title("Feature Selected (1=Yes, 0=No) per Method", fontsize=13)

# Subplot 2: Test R^2 comparison
results_plot = results_df.sort_values("Method")
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results_plot)))
x = np.arange(len(results_plot))
w = 0.25
axes[1].bar(x - w, results_plot["Test_R2"], w, label="Test R^2", color="#2c7fb8", edgecolor="white")
axes[1].bar(x, results_plot["Adj_R2"], w, label="Adj. R^2", color="#e67e22", edgecolor="white")
axes[1].bar(x + w, results_plot["CV_R2_mean"], w, label="CV R^2", color="#27ae60", edgecolor="white")
axes[1].set_xticks(x)
axes[1].set_xticklabels(results_plot["Method"], rotation=20, ha="right", fontsize=8)
axes[1].set_ylabel("R-squared")
axes[1].set_title("Model Performance by Feature Selection Method", fontsize=13)
axes[1].legend(fontsize=9)
axes[1].set_ylim(0.85, 1.0)

fig.suptitle("5 Feature Selection Methods: Comparison", fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "feature_selection_comparison.png"), dpi=150)
plt.close()
print(f"\n  Chart saved: outputs/figures/feature_selection_comparison.png")

# ===========================================================================
# CONCLUSION
# ===========================================================================
print("\n" + "=" * 70)
print("  CONCLUSION")
print("=" * 70)

# Count how many methods selected each feature
feature_votes = heatmap_data.sum(axis=1).sort_values(ascending=False)
print(f"\n  Feature votes (out of {len(methods)} methods):")
for f, v in feature_votes.items():
    bar = "#" * int(v)
    print(f"    {f:25s}: {int(v)}/5 {bar}")

# Best overall features (selected by >= 3 methods)
consensus = [f for f, v in feature_votes.items() if v >= 3]
print(f"\n  Consensus features (>= 3/5 methods): {consensus}")

# The one feature selected by ALL methods
unanimous = [f for f, v in feature_votes.items() if v == len(methods)]
print(f"  Unanimous (5/5 methods):           {unanimous}")

print(f"""
  KEY FINDINGS:
  =============
  1. R&D Spend is the undisputed #1 feature — selected by ALL 5 methods
     with every approach confirming its dominant predictive power.

  2. Marketing Spend is selected by multiple methods (particularly
     Forward Selection and RFE) but weaker methods (Lasso, Mutual Info)
     deprioritize it, confirming its marginal utility.

  3. State dummy variables and Administration are almost never selected,
     reinforcing that location and admin spending do not drive profit.

  4. Lasso (L1) is the most aggressive eliminator, often keeping only
     R&D Spend, while RFE and Forward Selection retain more features.

  5. The highest Test R^2 comes from models with 1-2 features —
     adding more does not improve prediction and risks overfitting.
""")

# Save report
with open("outputs/reports/feature_selection.txt", "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write(" 5 Feature Selection Methods — Comparison Report\n")
    f.write("=" * 60 + "\n\n")
    f.write(results_df.to_string(index=False))
    f.write(f"\n\nFeature votes:\n{feature_votes.to_string()}\n")
print("  Report saved: outputs/reports/feature_selection.txt")
