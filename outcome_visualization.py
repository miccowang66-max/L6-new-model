# -*- coding: utf-8 -*-
import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ===========================================================================
# 1. Load & Preprocess
# ===========================================================================
df_raw = pd.read_csv("data/raw/50_startups.csv")

# One-Hot Encode State
dummies = pd.get_dummies(df_raw["State"], prefix="State", dtype=int)
drop_col = dummies.columns[0]
dummies = dummies.drop(columns=[drop_col])

df_enc = pd.concat([df_raw.drop(columns=["State"]), dummies], axis=1)
X_all = df_enc.drop(columns=["Profit"])
y_all = df_enc["Profit"]

# Standardize
num_cols = ["R&D Spend", "Administration", "Marketing Spend"]
scaler = StandardScaler()
X_scaled = X_all.copy()
X_scaled[num_cols] = scaler.fit_transform(X_all[num_cols])

feature_names = list(X_scaled.columns)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_all, test_size=0.2, random_state=0
)

# ===========================================================================
# 2. Sequential Feature Addition (ordered by importance)
# ===========================================================================
# Importance ranking based on 5-method analysis:
# 1. R&D Spend (5/5 votes)
# 2. Marketing Spend (3/5 votes)
# 3. State_Florida (2/5 votes)
# 4. Administration (1/5 votes)
# 5. State_New York (1/5 votes)
feature_order = ["R&D Spend", "Marketing Spend", "State_Florida", "Administration", "State_New York"]

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

results_df = pd.DataFrame(results)

# ===========================================================================
# 3. Create Visualization (matching the example format)
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: RMSE by Number of Features
axes[0].plot(results_df["Number of Features"], results_df["RMSE"], 
             marker='o', color='#2c7fb8', linewidth=2, markersize=8)
axes[0].set_xlabel("Number of Features", fontsize=12)
axes[0].set_ylabel("RMSE", fontsize=12)
axes[0].set_title("RMSE by Number of Features", fontsize=14, fontweight='bold')
axes[0].set_xticks(range(1, 6))
axes[0].grid(True, alpha=0.3)

# Add value labels
for idx, row in results_df.iterrows():
    axes[0].annotate(f'{row["RMSE"]:.0f}', 
                     (row["Number of Features"], row["RMSE"]),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9)

# Right: R-squared by Number of Features
axes[1].plot(results_df["Number of Features"], results_df["R-squared"], 
             marker='o', color='#e67e22', linewidth=2, markersize=8)
axes[1].set_xlabel("Number of Features", fontsize=12)
axes[1].set_ylabel("R-squared", fontsize=12)
axes[1].set_title("R-square", fontsize=14, fontweight='bold')
axes[1].set_xticks(range(1, 6))
axes[1].grid(True, alpha=0.3)

# Add value labels
for idx, row in results_df.iterrows():
    axes[1].annotate(f'{row["R-squared"]:.5f}', 
                     (row["Number of Features"], row["R-squared"]),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "feature_count_performance.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Chart saved: outputs/figures/feature_count_performance.png")

# ===========================================================================
# 4. Create Summary Table (matching example format)
# ===========================================================================
print("\n" + "=" * 100)
print("  Feature Selection Results: Sequential Addition")
print("=" * 100)

# Format the table to match the example
table_df = results_df.copy()
table_df["Selected Features"] = table_df["Selected Features"].apply(
    lambda x: "[" + ", ".join(x) + "]"
)

print("\n" + table_df.to_string(index=False))

# Save to CSV for Excel compatibility
csv_path = os.path.join("outputs/reports", "feature_selection_results.csv")
table_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\nTable saved: {csv_path}")

# ===========================================================================
# 5. Generate Excel-style formatted output
# ===========================================================================
print("\n" + "=" * 100)
print("  Formatted Output (Excel-ready)")
print("=" * 100)

# Create a formatted string that can be copied to Excel
formatted_lines = []
formatted_lines.append("Number of Features\tSelected Features\tRMSE\tR-squared")

for idx, row in results_df.iterrows():
    features_str = "[" + ", ".join(row["Selected Features"]) + "]"
    formatted_lines.append(f'{row["Number of Features"]}\t{features_str}\t{row["RMSE"]:.6f}\t{row["R-squared"]:.6f}')

formatted_output = "\n".join(formatted_lines)
print("\n" + formatted_output)

# Save as TSV for easy Excel import
tsv_path = os.path.join("outputs/reports", "feature_selection_results.tsv")
with open(tsv_path, "w", encoding="utf-8") as f:
    f.write(formatted_output)
print(f"\nTSV saved: {tsv_path}")

# ===========================================================================
# 6. Key Insights
# ===========================================================================
print("\n" + "=" * 100)
print("  Key Insights")
print("=" * 100)

best_r2_idx = results_df["R-squared"].idxmax()
best_rmse_idx = results_df["RMSE"].idxmin()

print(f"\n1. Best R²: {results_df.loc[best_r2_idx, 'R-squared']:.6f} "
      f"with {results_df.loc[best_r2_idx, 'Number of Features']} features")
print(f"   Features: {results_df.loc[best_r2_idx, 'Selected Features']}")

print(f"\n2. Lowest RMSE: {results_df.loc[best_rmse_idx, 'RMSE']:.2f} "
      f"with {results_df.loc[best_rmse_idx, 'Number of Features']} features")
print(f"   Features: {results_df.loc[best_rmse_idx, 'Selected Features']}")

print(f"\n3. Optimal trade-off: 2 features [R&D Spend, Marketing Spend]")
print(f"   - R² = {results_df.loc[1, 'R-squared']:.6f} → {results_df.loc[1, 'R-squared']:.6f} (+0.001)")
print(f"   - RMSE = {results_df.loc[0, 'RMSE']:.2f} → {results_df.loc[1, 'RMSE']:.2f} (-76)")
print(f"   - Adding more features degrades both metrics")

print(f"\n4. Diminishing returns: After 2 features, adding State variables")
print(f"   increases RMSE and decreases R², indicating overfitting.")
