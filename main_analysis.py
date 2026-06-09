#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
 50_Startups 多元線性迴歸分析 — 逐步淘汰法（Backward Elimination）
 預測目標：Profit（利潤）
================================================================================
 此腳本涵蓋完整 ML 管線：
   1. 相關性分析與視覺化
   2. 資料前處理（One-Hot Encoding + 虛擬變數陷阱處理）
   3. 逐步淘汰法（Backward Elimination via P-value）
   4. 模型評估（R^2, Adj. R^2, RMSE, 並輸出最終方程式）
================================================================================
"""

import os, sys, warnings, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ----------------------------- 全域設定 ----------------------------------------
RAW_DATA = "data/raw/50_startups.csv"
PROCESSED_DIR = "data/processed"
FIGURES_DIR = "outputs/figures"
MODEL_DIR = "outputs/models"
REPORTS_DIR = "outputs/reports"
TARGET = "Profit"
RANDOM_STATE = 0
TEST_SIZE = 0.2

_ = [os.makedirs(d, exist_ok=True) for d in (PROCESSED_DIR, FIGURES_DIR, MODEL_DIR, REPORTS_DIR)]

# 中文字型設定 (Windows)
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


# ==============================================================================
#  第一部分：相關性分析
# ==============================================================================
print("=" * 70)
print(" 第一部分：相關性分析（Correlation Analysis）")
print("=" * 70)

# --- 1.1 載入原始資料（唯讀） ------------------------------------------------
print("\n[1.1] 載入原始資料...")
df_raw = pd.read_csv(RAW_DATA)
print(f"      資料形狀 (row, col): {df_raw.shape}")
print(f"      欄位名稱: {list(df_raw.columns)}")
print("\n      前 5 筆資料：")
print(df_raw.head().to_string(index=False))
print(f"\n      各欄位遺失值統計：\n{df_raw.isnull().sum()}")
print(f"\n      數值欄位統計摘要：\n{df_raw.describe().to_string()}")

# --- 1.2 計算相關係數矩陣 ----------------------------------------------------
print("\n[1.2] 計算數值特徵與 Profit 的相關係數...")

num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df_raw[num_cols].corr()
profit_corr = corr_matrix[TARGET].drop(TARGET).sort_values(ascending=False)

print(f"\n      各特徵與 Profit 的 Pearson 相關係數（降冪排序）：")
for feat, val in profit_corr.items():
    bar = "█" * int(abs(val) * 50) if abs(val) * 50 > 0 else ""
    direction = "+" if val >= 0 else "-"
    print(f"        {feat:<25s}  {direction}{abs(val):.4f}  {bar}")

# --- 1.3 視覺化：相關性熱力圖 ------------------------------------------------
print("\n[1.3] 繪製相關性熱力圖 (corr_heatmap.png)...")
fig, ax = plt.subplots(figsize=(10, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            mask=mask, linewidths=1.2, square=True, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("特徵間 Pearson 相關係數矩陣（熱力圖）", fontsize=15, pad=15)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "corr_heatmap.png"), dpi=150)
plt.close()
print("      已儲存 -> outputs/figures/corr_heatmap.png")

# --- 1.4 視覺化：配對散佈圖 -------------------------------------------------
print("[1.4] 繪製特徵 vs Profit 散佈圖 (scatter_features.png)...")
num_features = [c for c in num_cols if c != TARGET]
fig, axes = plt.subplots(1, len(num_features), figsize=(16, 5))
for i, feat in enumerate(num_features):
    axes[i].scatter(df_raw[feat], df_raw[TARGET], alpha=0.7,
                    c="#2c7fb8", edgecolors="white", linewidth=0.5, s=60)
    axes[i].set_xlabel(feat, fontsize=11)
    axes[i].set_ylabel(TARGET, fontsize=11)
    axes[i].set_title(f"{feat} vs {TARGET}\nr = {df_raw[feat].corr(df_raw[TARGET]):.4f}",
                      fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "scatter_features.png"), dpi=150)
plt.close()
print("      已儲存 -> outputs/figures/scatter_features.png")

# --- 1.5 State 對 Profit 的箱型圖 -------------------------------------------
print("[1.5] 繪製 State vs Profit 箱型圖 (boxplot_state.png)...")
fig, ax = plt.subplots(figsize=(8, 5))
order_stats = df_raw.groupby("State")[TARGET].median().sort_values().index.tolist()
sns.boxplot(x="State", y=TARGET, data=df_raw, order=order_stats,
            palette="Set2", ax=ax)
sns.stripplot(x="State", y=TARGET, data=df_raw, order=order_stats,
              color="black", alpha=0.4, size=4, ax=ax)
ax.set_title("不同 State 的 Profit 分佈", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "boxplot_state.png"), dpi=150)
plt.close()
print("      已儲存 -> outputs/figures/boxplot_state.png")

print("\n>>> 第一部分完成：相關性分析顯示 R&D Spend 與 Profit 高度正相關，"
      "而 Administration 相關性極低。")


# ==============================================================================
#  第二部分：資料前處理（One-Hot Encoding + 避免虛擬變數陷阱）
# ==============================================================================
print("\n" + "=" * 70)
print(" 第二部分：資料前處理（Data Preparation）")
print("=" * 70)

# --- 2.1 One-Hot Encoding ----------------------------------------------------
print("\n[2.1] 對 State 欄位進行 One-Hot Encoding...")
df = df_raw.copy()   # 從不修改原始 df_raw

state_dummies = pd.get_dummies(df["State"], prefix="State", dtype=int)
print(f"      原始 State 類別: {df['State'].unique().tolist()}")
print(f"      產生的虛擬變數: {state_dummies.columns.tolist()}")

# --- 2.2 避開虛擬變數陷阱 (Dummy Variable Trap) -------------------------------
print("\n[2.2] 刪除第一個虛擬變數以避開 Dummy Variable Trap...")
drop_col = state_dummies.columns[0]   # 第 k 個類別作為基準 (baseline)
state_dummies = state_dummies.drop(columns=[drop_col])
print(f"      已刪除: '{drop_col}'（作為基準類別）")
print(f"      保留:   {state_dummies.columns.tolist()}")
print("      說明: 保留 k-1 個 dummy 即可完整表達 k 個類別的訊息。")

# --- 2.3 合併特徵矩陣 --------------------------------------------------------
print("\n[2.3] 合併數值特徵與虛擬變數...")
df = pd.concat([df.drop(columns=["State"]), state_dummies], axis=1)
print(f"      合併後特徵矩陣形狀: {df.shape}")
print(f"      欄位: {df.columns.tolist()}")

# --- 2.4 分離 X 與 y ---------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]
print(f"\n[2.4] 特徵矩陣 X ({X.shape[1]} 個特徵) 與目標向量 y (長度 {len(y)})")

# --- 2.5 標準化 --------------------------------------------------------------
print("\n[2.5] 特徵標準化 (StandardScaler)...")
# 僅對數值特徵做標準化（dummy 變數已經是 0/1，不需要標準化）
num_features_now = [c for c in X.columns if c not in state_dummies.columns]
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[num_features_now] = scaler.fit_transform(X[num_features_now])
print(f"      已標準化特徵: {num_features_now}")
print("      標準化後前 3 筆：")
print(X_scaled.head(3).to_string(index=False))

# --- 2.6 訓練集/測試集分割 ---------------------------------------------------
print("\n[2.6] 切割訓練集 (80%) 與測試集 (20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"      訓練集 X: {X_train.shape}, y: {y_train.shape}")
print(f"      測試集 X: {X_test.shape}, y: {y_test.shape}")

# --- 2.7 保存處理後資料 ------------------------------------------------------
X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)
print(f"\n[2.7] 已保存處理後資料至 {PROCESSED_DIR}/")

print("\n>>> 第二部分完成：One-Hot Encoding 完成，已避開虛擬變數陷阱。")


# ==============================================================================
#  第三部分：逐步淘汰法（Backward Elimination）
# ==============================================================================
print("\n" + "=" * 70)
print(" 第三部分：逐步淘汰法（Backward Elimination）")
print("=" * 70)

# statsmodels 需要在設計矩陣前手動加入截距項 (intercept)
# Backward Elimination 流程：
#   1. 選擇一個顯著水準 alpha（例如 0.05）
#   2. 用所有特徵擬合模型
#   3. 找出 P-value 最高的特徵
#   4. 若 P-value > alpha，則剔除該特徵，回到步驟 2
#   5. 若所有特徵 P-value <= alpha，則停止

SIGNIFICANCE_LEVEL = 0.05  # 顯著水準 alpha

def backward_elimination(X: pd.DataFrame, y: pd.Series, sl: float = 0.05,
                         verbose: bool = True) -> list:
    """
    逐步淘汰法（Backward Elimination）

    參數:
        X:  特徵矩陣 DataFrame（標準化後，不含截距項）
        y:  目標向量 Series
        sl: 顯著水準 (significance level)，預設 0.05
        verbose: 是否輸出淘汰過程

    回傳:
        selected_features: 最終保留的特徵名稱列表
    """
    X_be = X.copy()
    # 1) 手動加入截距項 (const = 1.0)
    X_be = sm.add_constant(X_be, has_constant="add")
    features = list(X.columns)

    iteration = 0
    removed_features = []

    while True:
        iteration += 1
        # 2) 用 OLS 擬合
        model = sm.OLS(y, X_be.astype(float)).fit()
        pvalues = model.pvalues.drop("const", errors="ignore")
        max_pval = pvalues.max()
        max_pval_feature = pvalues.idxmax()

        if verbose:
            print(f"\n      -- 第 {iteration} 輪 --")
            print(f"        模型特徵: {list(X_be.columns)}")
            print(f"        各特徵 P-value:\n{pvalues.to_string()}")
            print(f"        Max P-value: {max_pval:.6f} ({max_pval_feature})")

        # 3) 判斷是否淘汰
        if max_pval > sl:
            removed_features.append((max_pval_feature, max_pval))
            if verbose:
                print(f"        >> 淘汰 '{max_pval_feature}' (P = {max_pval:.6f} > alpha = {sl})")
            # 從設計矩陣中刪除該特徵
            X_be = X_be.drop(columns=[max_pval_feature])
            features.remove(max_pval_feature)
        else:
            if verbose:
                print(f"        >> 停止：所有 P-value <= alpha = {sl}")
            break

    selected_features = [f for f in X.columns if f in features]
    return selected_features


# --- 3.1 執行逐步淘汰法 ------------------------------------------------------
print(f"\n[3.1] 開始 Backward Elimination（alpha = {SIGNIFICANCE_LEVEL}）...")
print(f"      初始特徵數: {X_train.shape[1]}")
print(f"      初始特徵:   {list(X_train.columns)}")

selected = backward_elimination(X_train, y_train, sl=SIGNIFICANCE_LEVEL)
print(f"\n[3.2] 最終保留特徵 ({len(selected)} 個): {selected}")

# 比對初始特徵與最終特徵
removed = [c for c in X_train.columns if c not in selected]
print(f"      被淘汰特徵 ({len(removed)} 個): {removed}")


# ==============================================================================
#  第四部分：模型訓練與評估
# ==============================================================================
print("\n" + "=" * 70)
print(" 第四部分：最終模型評估")
print("=" * 70)

# --- 4.1 用保留的特徵重新訓練最終模型 ---------------------------------------
print(f"\n[4.1] 使用最終特徵組合訓練模型...")
print(f"      保留特徵: {selected}")

X_train_final = X_train[selected]
X_test_final = X_test[selected]

# statsmodels OLS（完整統計摘要）
X_train_sm = sm.add_constant(X_train_final.astype(float))
final_model_sm = sm.OLS(y_train, X_train_sm).fit()

print("\n[4.2] statsmodels OLS 模型摘要：")
print("-" * 70)
print(final_model_sm.summary())
print("-" * 70)

# --- 4.3 測試集表現 ---------------------------------------------------------
print("\n[4.3] 測試集表現評估...")

# scikit-learn LinearRegression（與 statsmodels 對照）
lr_final = LinearRegression()
lr_final.fit(X_train_final, y_train)
y_pred = lr_final.predict(X_test_final)

# 評估指標
n = X_test_final.shape[0]   # 測試集樣本數
p = X_test_final.shape[1]   # 特徵數量

r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n      測試集樣本數 (n):          {n}")
print(f"      特徵數 (p):                {p}")
print(f"      R-squared Score:           {r2:.6f}")
print(f"      Adjusted R-squared:        {adj_r2:.6f}")
print(f"      Mean Absolute Error:       {mae:.4f}  (平均絕對誤差)")
print(f"      Root Mean Squared Error:   {rmse:.4f}  (均方根誤差)")

# 各點預測 vs 實際
print(f"\n      預測 vs 實際（前 10 筆測試樣本）:")
pred_vs_actual = pd.DataFrame({
    "實際 Profit": y_test.values,
    "預測 Profit": y_pred,
    "殘差 (Residual)": y_test.values - y_pred
}).reset_index(drop=True)
print(pred_vs_actual.head(10).to_string())

# --- 4.4 預測 vs 實際 視覺化 ------------------------------------------------
print("\n[4.4] 繪製預測 vs 實際圖 (pred_vs_actual.png)...")
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred, alpha=0.7, color="#2c7fb8", edgecolors="white",
           linewidth=0.5, s=80, label="測試樣本")
# 完美預測線
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2,
        label="完美預測線 (y = x)")
ax.set_xlabel("實際 Profit", fontsize=12)
ax.set_ylabel("預測 Profit", fontsize=12)
ax.set_title(f"最終模型預測 vs 實際\nR^2 = {r2:.4f}, Adj. R^2 = {adj_r2:.4f}, RMSE = {rmse:.2f}",
             fontsize=13)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "pred_vs_actual.png"), dpi=150)
plt.close()
print("      已儲存 -> outputs/figures/pred_vs_actual.png")

# --- 4.5 殘差圖 ------------------------------------------------------------
print("[4.5] 繪製殘差分佈圖 (residuals.png)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 殘差 vs 預測值
axes[0].scatter(y_pred, y_test.values - y_pred, alpha=0.7,
                color="#2c7fb8", edgecolors="white", linewidth=0.5, s=70)
axes[0].axhline(y=0, color="red", linestyle="--", linewidth=1.5)
axes[0].set_xlabel("預測值", fontsize=12)
axes[0].set_ylabel("殘差 (Actual - Predicted)", fontsize=12)
axes[0].set_title("殘差 vs 預測值", fontsize=13)

# 殘差直方圖
axes[1].hist(y_test.values - y_pred, bins=12, color="#2c7fb8", edgecolor="white",
             alpha=0.8)
axes[1].axvline(x=0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("殘差", fontsize=12)
axes[1].set_ylabel("頻率", fontsize=12)
axes[1].set_title("殘差直方圖（應接近常態分佈）", fontsize=13)

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "residuals.png"), dpi=150)
plt.close()
print("      已儲存 -> outputs/figures/residuals.png")

# --- 4.6 輸出最終迴歸方程式 -------------------------------------------------
print("\n[4.6] 最終迴歸方程式：")
coef = final_model_sm.params
intercept = coef["const"]
equation_parts = [f"Profit = {intercept:.4f}"]
for feat in selected:
    c = coef[feat]
    equation_parts.append(f" ({c:+.4f}) * {feat}")
equation = "".join(equation_parts)
print(f"\n      {equation}")

# --- 4.7 保存評估報告 -------------------------------------------------------
report_path = os.path.join(REPORTS_DIR, "metrics.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write(" 50_Startups 多元線性迴歸 — 模型評估報告\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"最終保留特徵 ({len(selected)} 個): {selected}\n")
    f.write(f"被淘汰特徵: {removed}\n")
    f.write(f"顯著水準 alpha: {SIGNIFICANCE_LEVEL}\n\n")
    f.write(f"測試集 R-squared Score:        {r2:.6f}\n")
    f.write(f"測試集 Adjusted R-squared:     {adj_r2:.6f}\n")
    f.write(f"測試集 MAE (平均絕對誤差):    {mae:.4f}\n")
    f.write(f"測試集 RMSE (均方根誤差):    {rmse:.4f}\n\n")
    f.write(f"迴歸方程式:\n{equation}\n\n")
    f.write(str(final_model_sm.summary()))
print(f"      已儲存完整報告 -> {report_path}")

# ==============================================================================
#  最終摘要
# ==============================================================================
print("\n" + "=" * 70)
print(" 分析完成 — 最終摘要")
print("=" * 70)
print(f"""
   [*] 初始特徵:         R&D Spend, Administration, Marketing Spend, State
   [>] 前處理後特徵數:     {X.shape[1]}
   [v] 逐步淘汰後保留:     {selected}
   [x] 遭淘汰特徵:         {removed}
   -------------------------------------------------
   [+] 最終模型測試集績效:
       R^2  = {r2:.6f}
       Adj. R^2 = {adj_r2:.6f}
       MAE = {mae:.4f}
       RMSE = {rmse:.4f}
   -------------------------------------------------
   解釋：{f"最終模型解釋了約 {r2*100:.1f}% 的 Profit 變異量。"
         if len(removed) > 0 else ""}
{textwrap.fill(
    f"逐步淘汰法發現 '{removed}' 對 Profit 的預測不具統計顯著性（P-value > alpha = {SIGNIFICANCE_LEVEL}），"
    f"因此被排除。最終以 {len(selected)} 個特徵建構的模型具有優秀的預測能力。",
    width=65) if removed else ""}
""")
