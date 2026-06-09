# L6 Crisp-RD2 — 技術規格白皮書

## Multiple Linear Regression Pipeline with 5 Feature Selection Methods

---

**版本**：1.0  
**日期**：2026-06-09  
**作者**：miccowang66-max  
**授權**：MIT License  

---

## 目錄

1. [專案概述](#1-專案概述)
2. [系統架構](#2-系統架構)
3. [資料規格](#3-資料規格)
4. [資料前處理管線](#4-資料前處理管線)
5. [特徵選取方法論](#5-特徵選取方法論)
6. [模型實作細節](#6-模型實作細節)
7. [實驗結果](#7-實驗結果)
8. [模型診斷](#8-模型診斷)
9. [部署架構](#9-部署架構)
10. [附錄](#10-附錄)

---

## 1. 專案概述

### 1.1 背景

本專案使用 50 間新創公司的營運數據，建構多元線性迴歸（Multiple Linear Regression）模型來預測公司利潤（Profit）。目標是識別哪些營運指標對獲利能力具有統計顯著性，並建立具備高度解釋力的精簡預測模型。

### 1.2 目標

| 目標 | 描述 |
|------|------|
| **特徵重要性排序** | 使用 5 種不同方法交叉驗證各特徵的預測貢獻度 |
| **最佳特徵組合** | 找出能最大化 R² 同時保持模型簡潔的特徵數量 |
| **穩健性驗證** | 透過離群值移除、Box-Cox 轉換、Huber 迴歸驗證模型穩定性 |
| **可重現管線** | 建立可複用的 ML 管線，封裝為 Agent Skill |

### 1.3 核心問題

> 哪些營運指標（R&D 投入、行銷支出、行政支出、地理位置）能顯著預測新創公司的利潤？是否存在一個最精簡的特徵子集能達到最優預測效果？

### 1.4 技術棧

| 類別 | 技術 | 版本要求 |
|------|------|---------|
| 語言 | Python | ≥ 3.10 |
| 資料處理 | pandas, numpy | ≥ 2.0 / ≥ 1.24 |
| 統計建模 | statsmodels | ≥ 0.14 |
| 機器學習 | scikit-learn | ≥ 1.3 |
| 視覺化 | matplotlib, seaborn | ≥ 3.7 / ≥ 0.12 |
| 科學計算 | scipy | ≥ 1.10 |

---

## 2. 系統架構

### 2.1 管線流程

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   Stage 1    │────▶│     Stage 2      │────▶│    Stage 3      │────▶│     Stage 4      │
│ Data Display │     │  Data Preparation │     │ 5-Method FS     │     │ Model Evaluation │
│ (READ-ONLY)  │     │   (Consolidated)  │     │ + Training      │     │ + Refinement     │
└──────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────┘
```

### 2.2 目錄結構

```
L6-new-model/
├── design.md                          # 架構設計文件（單一事實來源）
├── requirements.txt                   # Python 相依套件
├── main_analysis.py                   # Stage 1-4 完整管線
├── feature_selection.py               # 5 種特徵選取方法比較
├── refined_models.py                  # 離群值移除 + Box-Cox + Huber
├── outcome_visualization.py           # 循序特徵新增圖表
├── method_comparison_charts.py        # 5 方法比較儀表板
├── supplement_analysis.py             # VIF、CV、Cook's D 診斷
├── index.html                         # GitHub Pages 儀表板
├── data/
│   ├── raw/                           # [唯讀] 原始資料
│   │   └── 50_startups.csv
│   └── processed/                     # [唯寫] 前處理後資料
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── outputs/
│   ├── figures/                       # 11 張 PNG 圖表
│   └── reports/                       # 6 份文字/CSV 報告
├── .github/workflows/
│   └── deploy.yml                     # GitHub Pages CI/CD
└── .opencode/skills/
    ├── ml-regression-pipeline/         # ML 管線 Skill
    └── skill-creator/                  # Skill 建立工具
```

### 2.3 設計原則

| 原則 | 描述 |
|------|------|
| **唯讀資料層** | `data/raw/` 中任何檔案不得被修改或覆寫 |
| **集中式前處理** | 所有資料轉換（編碼、標準化、分割）集中在單一模組 |
| **模組分離** | 每個 `src/` 模組僅負責單一職責 |
| **可重現性** | 固定 `random_state=0`，所有分割可完全重現 |
| **Skill 封裝** | 完整管線封裝為可複用的 OpenCode Agent Skill |

---

## 3. 資料規格

### 3.1 資料來源

| 屬性 | 值 |
|------|-----|
| 檔案名稱 | `50_startups.csv` |
| 樣本數 | 50 筆 |
| 來源 | Kaggle / 公開學術資料集 |

### 3.2 欄位結構

| 欄位名稱 | 資料型態 | 描述 | 範圍 |
|----------|---------|------|------|
| R&D Spend | float64 | 研發支出（美元） | $0 – $165,349 |
| Administration | float64 | 行政支出（美元） | $51,283 – $182,646 |
| Marketing Spend | float64 | 行銷支出（美元） | $0 – $471,784 |
| State | object | 公司所在州 | California / Florida / New York |
| **Profit** | float64 | **利潤（目標變數，美元）** | $14,681 – $192,262 |

### 3.3 資料品質

| 檢查項目 | 結果 |
|----------|------|
| 缺失值 | 0 個（所有欄位完整） |
| 重複列 | 0 筆 |
| R&D Spend = 0 | 2 筆（潛在異常值） |
| Marketing Spend = 0 | 3 筆 |

### 3.4 數值統計摘要

| 欄位 | Mean | Std | Min | 25% | 50% | 75% | Max |
|------|------|-----|-----|------|------|------|------|
| R&D Spend | 73,722 | 45,902 | 0 | 39,936 | 73,051 | 101,603 | 165,349 |
| Administration | 121,345 | 28,018 | 51,283 | 103,731 | 122,700 | 144,842 | 182,646 |
| Marketing Spend | 211,025 | 122,290 | 0 | 129,300 | 212,716 | 299,469 | 471,784 |
| Profit | 112,013 | 40,306 | 14,681 | 90,139 | 107,978 | 139,766 | 192,262 |

---

## 4. 資料前處理管線

### 4.1 處理步驟（嚴格依序）

```
Step 1: 複製原始資料 ──▶ df = df_raw.copy()
Step 2: One-Hot Encoding ──▶ pd.get_dummies(df["State"], prefix="State", dtype=int)
Step 3: 避開虛擬變數陷阱 ──▶ 刪除第一個 dummy 欄位（基準類別：California）
Step 4: 分離特徵與目標 ──▶ X = df.drop(columns=["Profit"]), y = df["Profit"]
Step 5: 標準化數值特徵 ──▶ StandardScaler（僅套用於 R&D, Admin, Marketing）
Step 6: 訓練/測試分割 ──▶ train_test_split(test_size=0.2, random_state=0)
Step 7: 保存處理後資料 ──▶ CSV 輸出至 data/processed/
```

### 4.2 One-Hot Encoding 規格

| State 原始值 | 轉換後 |
|-------------|--------|
| California | 基準類別（被刪除，隱含為 0,0） |
| Florida | State_Florida = 1 |
| New York | State_New York = 1 |

> **設計決策**：僅保留 k-1 = 2 個虛擬變數，避免完整共線性（Dummy Variable Trap）。

### 4.3 標準化策略

- **套用對象**：僅數值特徵（R&D Spend, Administration, Marketing Spend）
- **不套用對象**：虛擬變數（State_Florida, State_New York）— 已為 0/1
- **方法**：`StandardScaler`（Z-score normalization）
- **擬合對象**：僅訓練集（`scaler.fit(X_train[num_cols])`）
- **轉換對象**：訓練集與測試集皆使用同一個 scaler（避免資料洩漏）

---

## 5. 特徵選取方法論

### 5.1 五種方法總覽

| # | 方法 | 類別 | 核心機制 | 超參數 |
|---|------|------|---------|--------|
| 1 | Backward Elimination | 統計檢定 | 從全模型逐步剔除 P-value > α 的變數 | α = 0.05 |
| 2 | Forward Selection | 統計檢定 | 從空模型逐步加入 P-value < α 的變數 | α = 0.05 |
| 3 | RFE (Recursive Feature Elimination) | 包裝法 | 迭代訓練並淘汰係數權重最低的特徵 | CV 選最優 n |
| 4 | Lasso Regression (L1) | 嵌入式 | L1 正則化自動將不重要係數壓縮至 0 | 5-fold CV 選 α |
| 5 | Mutual Information | 過濾法 | 衡量每個特徵與目標的非線性相依程度 | 閾值 = max(MI) × 0.2 |

### 5.2 方法 1 & 2：逐步迴歸

- **模型**：`statsmodels.OLS`（最小平方法）
- **顯著水準**：α = 0.05
- **終止條件**：
  - Backward：所有保留特徵的 P-value ≤ 0.05
  - Forward：沒有任何未選取特徵的 P-value < 0.05
- **評估指標**：P-value、R²、AIC

### 5.3 方法 3：RFE

- **基礎估計器**：`sklearn.linear_model.LinearRegression`
- **特徵數選擇**：遍歷 n = 1 至全部特徵數，以 5-fold CV R² 選最優 n
- **淘汰策略**：每輪移除 `coef_` 絕對值最小的特徵

### 5.4 方法 4：Lasso L1

- **正則化參數 α**：透過 `LassoCV` 以 5-fold CV 在 α ∈ [10⁻⁴, 10²] 中搜尋
- **最大迭代**：10,000
- **特徵選取規則**：| coefficient | > 10⁻⁵ 視為被選取

### 5.5 方法 5：Mutual Information

- **定義**：I(X;Y) = 衡量知道 X 後對 Y 的不確定性減少量
- **優勢**：能捕捉非線性關係（相較於 Pearson r）
- **閾值策略**：保留 MI ≥ MI_max × 20% 的特徵

### 5.6 投票機制

```
特徵得票數 = Σ(各方法是否選取該特徵)

共識特徵：得票數 ≥ 3/5（多數決）
全票特徵：得票數 = 5/5（無爭議）
```

---

## 6. 模型實作細節

### 6.1 基礎模型

- **演算法**：Ordinary Least Squares (OLS) Multiple Linear Regression
- **損失函數**：Σ(yᵢ - ŷᵢ)²
- **實作**：`statsmodels.OLS`（推論）+ `sklearn.LinearRegression`（預測）

### 6.2 精煉模型變體

| 變體 | 描述 | 觸發條件 |
|------|------|---------|
| Model A: Cleaned OLS | 移除 Cook's D 高影響點後重新訓練 | 任一點 Cook's D > 4/n |
| Model B: Box-Cox | 對目標變數 y 做 Box-Cox 轉換後建模 | 殘差 Omnibus p < 0.05 |
| Model C: Huber | 使用 Huber Loss 降低離群值影響 | 穩健性驗證需求 |
| Model D: No_RD Flag | 加入「零研發支出」二元旗標 | R&D = 0 樣本存在 |

### 6.3 評估指標

| 指標 | 公式 | 用途 |
|------|------|------|
| **R²** | 1 - SS_res / SS_tot | 解釋變異比例 |
| **Adjusted R²** | 1 - (1-R²)(n-1)/(n-p-1) | 懲罰特徵數量後的 R² |
| **MAE** | (1/n) Σ | yᵢ - ŷᵢ | | 平均絕對誤差（原始單位） |
| **RMSE** | √( (1/n) Σ (yᵢ - ŷᵢ)² ) | 均方根誤差（懲罰大誤差） |
| **5-fold CV R²** | 五折交叉驗證 R² 均值 ± 標準差 | 泛化能力評估 |

---

## 7. 實驗結果

### 7.1 相關性分析

| 特徵 | Pearson r vs Profit | 相關性強度 |
|------|-----|------|
| R&D Spend | 0.9729 | 極強正相關 |
| Marketing Spend | 0.7478 | 強正相關 |
| Administration | 0.2007 | 弱正相關 |

### 7.2 五種特徵選取方法比較

| 方法 | 選取特徵數 | 選取特徵 | 測試 R² | Adj. R² | 測試 RMSE |
|------|:---:|------|:---:|:---:|---:|
| Mutual Info (top) | 2 | R&D, Marketing | **0.9474** | 0.9324 | $8,198.80 |
| Backward Elimination | 1 | R&D | 0.9465 | **0.9398** | $8,274.87 |
| Forward Selection | 1 | R&D | 0.9465 | **0.9398** | $8,274.87 |
| RFE (CV-optimal) | 3 | R&D, Marketing, State_FL | 0.9451 | 0.9177 | $8,376.45 |
| Lasso L1 | 5 | All features | 0.9347 | 0.8531 | $9,137.99 |

### 7.3 特徵得票結果

| 特徵 | 得票數 | 狀態 |
|------|:---:|------|
| **R&D Spend** | **5/5** | 🏆 全票通過 — 無爭議核心預測因子 |
| **Marketing Spend** | 3/5 | ✅ 共識 — 邊際貢獻 |
| State_Florida | 2/5 | ⚠️ 弱證據 |
| Administration | 1/5 | ❌ 噪音 |
| State_New York | 1/5 | ❌ 噪音 |

### 7.4 循序特徵新增分析

| 特徵數 | 新增特徵 | RMSE | R² | 變化 |
|:---:|------|---:|---:|------|
| 1 | R&D Spend | $8,274.87 | 0.9465 | 基準 |
| **2** | **+ Marketing Spend** | **$8,198.80** | **0.9474** | ✅ **最佳** |
| 3 | + State_Florida | $8,376.45 | 0.9451 | ⚠️ 性能下降 |
| 4 | + Administration | $9,068.54 | 0.9357 | ❌ 過擬合 |
| 5 | + State_New York | $9,137.99 | 0.9347 | ❌ 過擬合 |

### 7.5 精煉模型比較

| 模型 | 描述 | 測試 R² | Adj. R² | MAE | RMSE |
|------|------|:---:|:---:|---:|---:|
| **Model A** | 移除 3 離群值 + OLS | **0.9601** | **0.9487** | $4,061 | $5,450 |
| Model B | Box-Cox 轉換 Profit | 0.9582 | 0.9530 | $3,677 | $5,575 |
| Model C | Huber 穩健迴歸 | 0.9568 | 0.9445 | $4,280 | $5,668 |

### 7.6 最終迴歸方程式

**標準化尺度**（Z-score）：
```
Profit = 111,199.31 + 38,698.52 × Z_R&D_Spend
```

**原始尺度**：
```
Profit = 49,032.00 + 0.854 × R&D_Spend
```

> 解讀：每增加 $1 研發支出，預期利潤增加約 $0.85。

**清洗離群值後（Model A）**：
```
Profit = 56,713.92 + 0.7649 × R&D_Spend + 0.0287 × Marketing_Spend
```

---

## 8. 模型診斷

### 8.1 基礎診斷（全資料集，n=50）

| 診斷指標 | 數值 | 判定 |
|----------|------|------|
| Cook's Distance (高影響點) | 3 個點 > 4/n | ⚠️ 需處理 |
| Omnibus Normality p-value | 0.0010 | ❌ 殘差非常態 |
| Jarque-Bera p-value | < 0.05 | ❌ 殘差非常態 |
| Durbin-Watson | 1.28 | ⚠️ 輕微正自相關 |
| VIF (max) | 2.47 (R&D Spend) | ✅ 無嚴重共線性 |

### 8.2 清洗後診斷（移除 3 離群值，n=47）

| 診斷指標 | 數值 | 判定 |
|----------|------|------|
| Omnibus Normality p-value | 0.831 | ✅ 殘差服從常態 |
| Jarque-Bera p-value | 0.773 | ✅ 殘差服從常態 |
| Durbin-Watson | 2.074 | ✅ 無自相關 |
| Marketing Spend P-value | 0.037 | ✅ 統計顯著 |
| Adjusted R² | 0.960 | ✅ 優秀 |

### 8.3 交叉驗證穩定性

| 特徵組合 | 5-fold CV R² Mean | Std |
|----------|:---:|:---:|
| R&D only | 0.9374 | 0.0373 |
| R&D + Marketing | 0.9389 | 0.0373 |
| R&D + Admin | 0.9304 | 0.0393 |

> 結果：加入 Marketing 或 Admin 未能顯著提升 CV R²，驗證了逐步淘汰法的決策。

### 8.4 各州 R&D → Profit 斜率一致性

| State | 斜率 | 州內 r | 樣本數 |
|------|:---:|:---:|:---:|
| California | 0.928 | 0.975 | 17 |
| New York | 0.813 | 0.976 | 17 |
| Florida | 0.813 | 0.970 | 16 |

> 三州斜率高度一致，確認 State 對 R&D→Profit 的關係沒有調節作用。

---

## 9. 部署架構

### 9.1 GitHub Pages 儀表板

- **觸發**：推送至 `master` 分支
- **建置**：GitHub Actions 自動部署靜態檔案
- **URL**：`https://miccowang66-max.github.io/L6-new-model/`
- **內容**：互動式 HTML 儀表板，內嵌全部 11 張分析圖表

### 9.2 CI/CD 管線

```yaml
觸發條件：push to master
├── checkout 原始碼
├── configure-pages（設定 Pages 環境）
├── upload-pages-artifact（上傳靜態檔案）
└── deploy-pages（部署至 GitHub Pages CDN）
```

### 9.3 Agent Skill 發佈

管線已封裝為 OpenCode Skill：
- 路徑：`.opencode/skills/ml-regression-pipeline/SKILL.md`
- 觸發條件：使用者提及 regression analysis、feature selection、backward elimination
- 可複用性：在其他專案中 Agent 可自動載入此 Skill 並按照相同流程執行

---

## 10. 附錄

### 10.1 產出圖表清單

| 檔名 | 描述 |
|------|------|
| `corr_heatmap.png` | Pearson 相關性熱力圖 |
| `scatter_features.png` | 各數值特徵 vs Profit 散佈圖 |
| `boxplot_state.png` | 各州 Profit 箱型圖 |
| `feature_count_performance.png` | 特徵數 vs RMSE/R² 雙線圖 |
| `feature_selection_comparison.png` | 5 方法特徵選取熱力圖 + R² 長條圖 |
| `method_comparison_rmse_r2.png` | 5 方法 RMSE / R² 水平長條圖 |
| `method_comparison_heatmap_performance.png` | 特徵選取熱力圖 + 特徵數 vs 性能散佈圖 |
| `method_comparison_dashboard.png` | 4 合 1 全覽儀表板 |
| `model_comparison.png` | 精煉模型 R²/MAE/RMSE 比較 |
| `pred_vs_actual.png` | 預測值 vs 實際值散佈圖 |
| `residuals.png` | 殘差 vs 預測值 + 殘差直方圖 |

### 10.2 報告清單

| 檔名 | 描述 |
|------|------|
| `metrics.txt` | 基礎 OLS 模型摘要 |
| `refined_models.txt` | 精煉模型比較報告 |
| `feature_selection.txt` | 5 方法特徵選取報告 |
| `feature_selection_results.csv` | 循序特徵新增數據（CSV） |
| `feature_selection_results.tsv` | 循序特徵新增數據（TSV，Excel 相容） |
| `method_comparison_results.csv` | 5 方法比較數據 |

### 10.3 詞彙表

| 術語 | 定義 |
|------|------|
| Dummy Variable Trap | 當 k 個虛擬變數全部保留時產生的完全共線性問題 |
| Backward Elimination | 從全模型開始，逐步移除 P-value 最高的不顯著變數 |
| VIF (Variance Inflation Factor) | 衡量共線性程度的指標，> 10 表示嚴重共線性 |
| Cook's Distance | 衡量單一資料點對整個迴歸模型影響力的指標 |
| Box-Cox Transformation | 將非常態分佈的變數轉換為近似常態的冪次轉換 |
| Omnibus Test | 檢定殘差是否服從常態分佈的綜合檢定 |
| Durbin-Watson | 檢測殘差是否存在一階自相關的統計量（理想值 ≈ 2.0） |

---

> **文件目的**：本白皮書作為 L6 Crisp-RD2 專案的技術規格文件，詳載了系統架構、方法論、實驗結果與部署細節。任何對專案的修改應先參照本文，確保一致性與可重現性。
