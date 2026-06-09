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
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ===========================================================================
# 0. Load & Preprocess
# ===========================================================================
df_raw = pd.read_csv("data/raw/50_startups.csv")
dummies = pd.get_dummies(df_raw["State"], prefix="State", dtype=int)
drop_col = dummies.columns[0]
dummies = dummies.drop(columns=[drop_col])
df_enc = pd.concat([df_raw.drop(columns=["State"]), dummies], axis=1)
X_all = df_enc.drop(columns=["Profit"])
y_all = df_enc["Profit"]

num_cols = ["R&D Spend", "Administration", "Marketing Spend"]
scaler = StandardScaler()
X_scaled = X_all.copy()
X_scaled[num_cols] = scaler.fit_transform(X_all[num_cols])
feature_names = list(X_scaled.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_all, test_size=0.2, random_state=0
)

# ===========================================================================
# Run all 5 methods and collect results
# ===========================================================================

# --- Method 1: Backward Elimination ---
def backward_elimination(X, y, sl=0.05):
    X_be = sm.add_constant(X.copy())
    features = list(X.columns)
    while True:
        model = sm.OLS(y, X_be.astype(float)).fit()
        pvalues = model.pvalues.drop("const", errors="ignore")
        max_pval = pvalues.max()
        if max_pval > sl:
            X_be = X_be.drop(columns=[pvalues.idxmax()])
            features.remove(pvalues.idxmax())
        else:
            break
    return [f for f in X.columns if f in features]

sel_back = backward_elimination(X_train, y_train)

# --- Method 2: Forward Selection ---
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

sel_forward = forward_selection(X_train, y_train)

# --- Method 3: RFE ---
cv = KFold(n_splits=5, shuffle=True, random_state=42)
best_n = 1
best_score = -999
for n_feat in range(1, X_train.shape[1] + 1):
    lr = LinearRegression()
    rfe = RFE(estimator=lr, n_features_to_select=n_feat)
    rfe.fit(X_train, y_train)
    scores = cross_val_score(lr, X_train.iloc[:, rfe.support_], y_train, cv=cv, scoring='r2')
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_n = n_feat
rfe = RFE(estimator=LinearRegression(), n_features_to_select=best_n)
rfe.fit(X_train, y_train)
sel_rfe = [f for f, s in zip(feature_names, rfe.support_) if s]

# --- Method 4: Lasso ---
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-4, 2, 50))
lasso_cv.fit(X_train, y_train)
sel_lasso = [f for f, c in zip(feature_names, lasso_cv.coef_) if abs(c) > 1e-5]

# --- Method 5: Mutual Information ---
mi = mutual_info_regression(X_train, y_train, random_state=42)
mi_threshold = mi.max() * 0.20
sel_mi = [f for f, m in zip(feature_names, mi) if m >= mi_threshold]

# ===========================================================================
# Evaluate all methods
# ===========================================================================
methods = {
    "Backward Elimination": sel_back,
    "Forward Selection": sel_forward,
    "RFE (CV-optimal)": sel_rfe,
    "Lasso L1": sel_lasso,
    "Mutual Info (top)": sel_mi,
}

results = []
for name, selected in methods.items():
    if len(selected) == 0:
        continue
    lr = LinearRegression().fit(X_train[selected], y_train)
    y_pred = lr.predict(X_test[selected])
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    n, p = len(y_test), len(selected)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    cv_scores = cross_val_score(LinearRegression(), X_train[selected], y_train, cv=5, scoring='r2')
    results.append({
        "Method": name,
        "n_Feat": len(selected),
        "Selected Features": "[" + ", ".join(selected) + "]",
        "RMSE": rmse,
        "R-squared": r2,
        "Adj. R-squared": adj_r2,
        "CV R^2": cv_scores.mean(),
    })

results_df = pd.DataFrame(results).sort_values("R-squared", ascending=False).reset_index(drop=True)

# ===========================================================================
# FIGURE 1: RMSE and R-squared bar charts (matching outcome example style)
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

method_labels = results_df["Method"].tolist()
x = np.arange(len(method_labels))
w = 0.6

# Left: RMSE
colors_rmse = plt.cm.Blues(np.linspace(0.4, 0.9, len(results_df)))
bars1 = axes[0].barh(x, results_df["RMSE"], w, color=colors_rmse, edgecolor="white", linewidth=0.8)
axes[0].set_yticks(x)
axes[0].set_yticklabels(method_labels, fontsize=10)
axes[0].invert_yaxis()
axes[0].set_xlabel("RMSE", fontsize=12)
axes[0].set_title("RMSE by Feature Selection Method", fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)
for i, row in results_df.iterrows():
    axes[0].text(row["RMSE"] + 30, i, f'{row["RMSE"]:.1f}', va='center', fontsize=9)

# Right: R-squared
colors_r2 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(results_df)))
bars2 = axes[1].barh(x, results_df["R-squared"], w, color=colors_r2, edgecolor="white", linewidth=0.8)
axes[1].set_yticks(x)
axes[1].set_yticklabels(method_labels, fontsize=10)
axes[1].invert_yaxis()
axes[1].set_xlabel("R-squared", fontsize=12)
axes[1].set_title("R-square by Feature Selection Method", fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)
axes[1].set_xlim(0.90, 0.96)
for i, row in results_df.iterrows():
    axes[1].text(row["R-squared"] + 0.0005, i, f'{row["R-squared"]:.5f}', va='center', fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "method_comparison_rmse_r2.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Chart 1 saved: outputs/figures/method_comparison_rmse_r2.png")

# ===========================================================================
# FIGURE 2: Feature votes heatmap + performance comparison
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Feature selection heatmap
feature_list = ["R&D Spend", "Marketing Spend", "State_Florida", "Administration", "State_New York"]
heatmap_data = pd.DataFrame(index=feature_list)
for method_name, selected in methods.items():
    short_name = method_name.split(" ")[0]
    heatmap_data[short_name] = [1 if f in selected else 0 for f in feature_list]

sns.heatmap(heatmap_data.T, annot=True, fmt="d", cmap="RdYlGn", cbar=False,
            linewidths=1.5, ax=axes[0], annot_kws={"size": 12})
axes[0].set_title("Feature Selected per Method\n(1=Yes, 0=No)", fontsize=13, fontweight='bold')
axes[0].set_ylabel("Method", fontsize=11)
axes[0].set_xlabel("Feature", fontsize=11)

# Right: n_Features vs RMSE / R² scatter
n_feats = results_df["n_Feat"].values
rmse_vals = results_df["RMSE"].values
r2_vals = results_df["R-squared"].values

ax2_left = axes[1]
ax2_right = ax2_left.twinx()

ax2_left.scatter(n_feats, rmse_vals, s=120, color='#2c7fb8', zorder=5, edgecolors='white', linewidth=1.5)
ax2_left.plot(n_feats, rmse_vals, color='#2c7fb8', linewidth=1.5, alpha=0.5)
ax2_left.set_xlabel("Number of Features", fontsize=12)
ax2_left.set_ylabel("RMSE", fontsize=12, color='#2c7fb8')
ax2_left.tick_params(axis='y', labelcolor='#2c7fb8')

ax2_right.scatter(n_feats, r2_vals, s=120, color='#e67e22', zorder=5, edgecolors='white', linewidth=1.5)
ax2_right.plot(n_feats, r2_vals, color='#e67e22', linewidth=1.5, alpha=0.5)
ax2_right.set_ylabel("R-squared", fontsize=12, color='#e67e22')
ax2_right.tick_params(axis='y', labelcolor='#e67e22')

axes[1].set_title("Features Count vs Performance", fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.3)

# Annotate points
for i in range(len(results_df)):
    axes[1].annotate(
        results_df.iloc[i]["Method"].split(" ")[0],
        (n_feats[i], rmse_vals[i]),
        textcoords="offset points", xytext=(8, 5), fontsize=8, color='#2c7fb8'
    )

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "method_comparison_heatmap_performance.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Chart 2 saved: outputs/figures/method_comparison_heatmap_performance.png")

# ===========================================================================
# FIGURE 3: Combined dashboard (all-in-one)
# ===========================================================================
fig = plt.figure(figsize=(20, 12))

# Top-left: RMSE bar
ax1 = fig.add_subplot(2, 2, 1)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(results_df)))[::-1]
bars = ax1.barh(range(len(results_df)), results_df["RMSE"], color=colors, edgecolor='white')
ax1.set_yticks(range(len(results_df)))
ax1.set_yticklabels(results_df["Method"], fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel("RMSE", fontsize=11)
ax1.set_title("RMSE Comparison", fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
for i, row in results_df.iterrows():
    ax1.text(row["RMSE"] + 20, i, f'{row["RMSE"]:.1f}', va='center', fontsize=9)

# Top-right: R² bar
ax2 = fig.add_subplot(2, 2, 2)
colors2 = plt.cm.YlOrRd(np.linspace(0.3, 0.85, len(results_df)))[::-1]
bars2 = ax2.barh(range(len(results_df)), results_df["R-squared"], color=colors2, edgecolor='white')
ax2.set_yticks(range(len(results_df)))
ax2.set_yticklabels(results_df["Method"], fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel("R-squared", fontsize=11)
ax2.set_title("R-squared Comparison", fontsize=13, fontweight='bold')
ax2.set_xlim(0.92, 0.96)
ax2.grid(axis='x', alpha=0.3)
for i, row in results_df.iterrows():
    ax2.text(row["R-squared"] + 0.0003, i, f'{row["R-squared"]:.5f}', va='center', fontsize=9)

# Bottom-left: Feature votes heatmap
ax3 = fig.add_subplot(2, 2, 3)
sns.heatmap(heatmap_data.T, annot=True, fmt="d", cmap="RdYlGn", cbar=False,
            linewidths=1.5, ax=ax3, annot_kws={"size": 11})
ax3.set_title("Feature Selection Votes\n(1=Selected, 0=Not)", fontsize=13, fontweight='bold')
ax3.set_ylabel("Method", fontsize=10)
ax3.set_xlabel("Feature", fontsize=10)

# Bottom-right: Summary table
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')
table_data = []
for i, row in results_df.iterrows():
    table_data.append([
        row["Method"],
        str(row["n_Feat"]),
        row["Selected Features"],
        f'{row["RMSE"]:.2f}',
        f'{row["R-squared"]:.5f}',
        f'{row["Adj. R-squared"]:.5f}',
    ])
col_labels = ["Method", "n_Feat", "Selected Features", "RMSE", "R²", "Adj. R²"]
table = ax4.table(cellText=table_data, colLabels=col_labels, loc='center',
                  cellLoc='center', colColours=['#2c3e50']*6)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.8)
# Color header
for j in range(6):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')
# Highlight best row (index 0 after sorting by R²)
for j in range(6):
    table[1, j].set_facecolor('#d5f5e3')
ax4.set_title("Summary Table (sorted by R²)", fontsize=13, fontweight='bold', pad=15)

fig.suptitle("5 Feature Selection Methods: Complete Comparison", fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "method_comparison_dashboard.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Chart 3 saved: outputs/figures/method_comparison_dashboard.png")

# ===========================================================================
# Print summary table
# ===========================================================================
print("\n" + "=" * 120)
print("  5 Feature Selection Methods — Full Comparison")
print("=" * 120)
print(results_df.to_string(index=False))

# Save CSV
csv_path = os.path.join("outputs/reports", "method_comparison_results.csv")
results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\nCSV saved: {csv_path}")

# Feature votes
feature_votes = heatmap_data.sum(axis=1).sort_values(ascending=False)
print(f"\n  Feature votes (out of {len(methods)} methods):")
for f, v in feature_votes.items():
    bar = "#" * int(v)
    print(f"    {f:25s}: {int(v)}/5 {bar}")
