# L6 Crisp-RD2 — 技術規格白皮書

## Multiple Linear Regression Pipeline with 5 Feature Selection Methods

---

**版本**：2.1
**日期**：2026-06-11
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
10. [快速入門與實作教學](#10-快速入門與實作教學)
11. [商業解讀與決策指南](#11-商業解讀與決策指南)
12. [常見問題與疑難排解（FAQ）](#12-常見問題與疑難排解faq)
13. [附錄](#13-附錄)
14. [變更日誌](#14-變更日誌-changelog)

---

## 1. 專案概述

### 1.1 背景

在現代商業環境中，新創公司的存活與成長高度依賴於資本配置的效率。風險投資者（Venture Capitalists）和企業策略師經常需要回答一個核心問題：**哪些營運支出能真正驅動利潤增長？** 這不僅關乎投資回報率的預測，更影響資源分配的戰略決策。

本專案使用一個經典的公開資料集——50 間美國新創公司的年度營運數據，建構多元線性迴歸（Multiple Linear Regression）模型來預測公司利潤（Profit）。該資料集包含三個數值型營運指標（R&D 研發支出、Administration 行政支出、Marketing 行銷支出）以及一個類別型變數（公司所在州別），目標變數為年度利潤。

這個問題具有高度的實務價值。對於早期投資者而言，若能準確預測一家新創公司在特定支出結構下的預期利潤，就能更有效地評估投資標的的潛力。對於新創公司經營者而言，了解各項支出對利潤的邊際貢獻，可以指導預算分配的優先順序。

### 1.2 CRISP-DM 方法論

本專案遵循 **CRISP-DM**（Cross-Industry Standard Process for Data Mining）方法論，這是資料探勘與機器學習領域最廣泛採用的產業標準流程。CRISP-DM 將資料科學專案劃分為六個階段，形成一個迭代循環：

```
┌──────────────┐     ┌──────────────┐
│  1. Business │────▶│  2. Data     │
│  Understanding│     │  Understanding│
└──────────────┘     └──────┬───────┘
       ▲                    │
       │            ┌───────▼───────┐
       │            │  3. Data      │
┌──────┴───────┐    │  Preparation  │
│  6.          │    └───────┬───────┘
│  Deployment  │            │
└──────▲───────┘    ┌───────▼───────┐
       │            │  4. Modeling  │
┌──────┴───────┐    └───────┬───────┘
│  5. Evaluation│◀───│               │
└──────────────┘    └───────────────┘
```

- **Phase 1 — Business Understanding**：定義商業目標。我們的目標是預測新創公司利潤，並識別最關鍵的營運指標。成功標準設定為模型 R² ≥ 0.90 且 RMSE ≤ $10,000。
- **Phase 2 — Data Understanding**：收集並探索 50 間新創公司的數據。檢查資料型態、缺失值、異常值，並透過相關性分析與視覺化來理解特徵之間的關係。
- **Phase 3 — Data Preparation**：建構資料前處理管線，包括 One-Hot Encoding、虛擬變數陷阱避免、特徵標準化，以及訓練集與測試集的分割（80/20 比例）。
- **Phase 4 — Modeling**：建構多元線性迴歸模型。除了基礎的 OLS 模型外，還實施了五種不同的特徵選取方法：Backward Elimination、Forward Selection、Recursive Feature Elimination、Lasso L1 正則化，以及 Mutual Information 過濾法。
- **Phase 5 — Evaluation**：使用多項指標評估模型性能（R²、Adjusted R²、MAE、RMSE、5-fold CV R²），並進行完整的殘差診斷（常態性檢定、Durbin-Watson 自相關檢定、VIF 共線性檢定、Cook's Distance 影響點分析）。
- **Phase 6 — Deployment**：將結果部署為雙平台——GitHub Pages 靜態儀表板與 Streamlit Cloud 互動式 Web 應用程式，並將完整管線封裝為可複用的 OpenCode Agent Skill。

### 1.3 目標

本專案設定了四個層級的目標，從探索性分析到可重現的工程管線：

| 目標 | 描述 |
|------|------|
| **特徵重要性排序** | 使用 5 種不同方法（統計檢定、包裝法、嵌入式、過濾法）交叉驗證各特徵的預測貢獻度，並以投票機制決定共識特徵。 |
| **最佳特徵組合** | 找出能最大化 R² 同時保持模型簡潔（Occam's Razor 原則）的特徵數量。透過循序特徵新增分析（Sequential Feature Addition）視覺化效能與複雜度的權衡。 |
| **穩健性驗證** | 透過離群值移除（基於 Cook's Distance）、Box-Cox 轉換（處理非常態殘差）、Huber 穩健迴歸（降低離群值影響）三種策略驗證模型在不同條件下的穩定性。 |
| **可重現管線** | 建立可複用的 ML 管線，封裝為 OpenCode Agent Skill，使其他資料科學家能在新專案中快速複製相同的方法論。 |

### 1.4 核心問題

本專案試圖回答三個層層遞進的研究問題：

> **Q1**：哪些營運指標（R&D 投入、行銷支出、行政支出、地理位置）能顯著預測新創公司的利潤？

> **Q2**：在這些指標中，是否存在一個最精簡的子集，能達到與全模型相當甚至更優的預測效果？

> **Q3**：不同的特徵選取方法（統計檢定 vs. 機器學習 vs. 資訊理論）是否會得出一致的結論？若不一致，應如何權衡？

這些問題的答案不僅具有學術價值，更對實務決策有直接指導意義。例如，若行政支出對利潤的預測不具統計顯著性，經營者就應該將更多注意力集中在 R&D 與行銷支出的配置上。

### 1.5 技術棧

以下為本專案使用的完整技術棧，依功能類別分組：

| 類別 | 技術 | 版本要求 | 用途 |
|------|------|---------|------|
| 語言 | Python | ≥ 3.10 | 核心程式語言 |
| 資料處理 | pandas | ≥ 2.0 | 資料框操作、合併、分組、統計摘要 |
| 數值計算 | numpy | ≥ 1.24 | 矩陣運算、數值穩定化 |
| 統計建模 | statsmodels | ≥ 0.14 | OLS 迴歸、P-value 計算、殘差診斷 |
| 機器學習 | scikit-learn | ≥ 1.3 | 線性迴歸、標準化、交叉驗證、Lasso、RFE |
| 靜態視覺化 | matplotlib | ≥ 3.7 | 基礎圖表繪製、自訂義樣式 |
| 統計視覺化 | seaborn | ≥ 0.12 | 熱力圖、箱型圖、散佈圖矩陣 |
| 互動式儀表板 | Streamlit | ≥ 1.55 | 互動式 CRISP-DM Web 應用程式 |
| 互動式圖表 | Plotly | ≥ 5.18 | Hover 互動、縮放、平移的折線圖 |
| 科學計算 | scipy | ≥ 1.10 | Box-Cox 轉換、統計檢定 |

### 1.6 先備知識

閱讀本白皮書前，建議具備以下背景知識：

- **機率與統計**：假設檢定（P-value、顯著水準 α）、相關係數（Pearson r）、殘差分析。
- **線性代數**：矩陣運算、最小平方法的正規方程式（Normal Equation）。
- **機器學習基礎**：偏差-變異權衡（Bias-Variance Tradeoff）、過擬合（Overfitting）、交叉驗證（Cross-Validation）。
- **Python 程式設計**：pandas DataFrame 操作、scikit-learn 管線 API、matplotlib 繪圖。

---

## 2. 系統架構

### 2.1 管線流程

本專案的資料處理與建模管線分為四個階段，每個階段的輸出成為下一個階段的輸入，形成一條嚴格有序的資料流：

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   Stage 1    │────▶│     Stage 2      │────▶│    Stage 3      │────▶│     Stage 4      │
│ Data Display │     │  Data Preparation │     │ 5-Method FS     │     │ Model Evaluation │
│ (READ-ONLY)  │     │   (Consolidated)  │     │ + Training      │     │ + Refinement     │
└──────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────┘
```

**Stage 1 — Data Display（資料展示）**：此階段為唯讀（READ-ONLY）操作，目標是對資料集進行初步的探索與理解，絕不修改原始檔案。具體任務包括：
- 載入 `data/raw/50_startups.csv` 並顯示資料形狀、欄位名稱、前五筆資料。
- 計算各欄位的缺失值數量與基本統計摘要（mean、std、min、quartiles、max）。
- 計算 Pearson 相關係數矩陣，並以熱力圖（heatmap）視覺化特徵之間的線性關係。
- 繪製各數值特徵與目標變數 Profit 的散佈圖（scatter plot），觀察線性趨勢的存在與否。
- 繪製不同州別的 Profit 箱型圖（boxplot），初步檢查地理位置是否對利潤有顯著影響。

**Stage 2 — Data Preparation（資料前處理 — 集中式）**：所有資料轉換操作集中在一個模組中執行，確保一致性與可重現性。步驟詳見第 4 章。

**Stage 3 — 5-Method Feature Selection + Training（特徵選取與模型訓練）**：此階段是本專案的核心，實施五種特徵選取方法並比較其結果。步驟詳見第 5 章與第 6 章。

**Stage 4 — Model Evaluation + Refinement（模型評估與精煉）**：對最終模型進行全面的診斷與優化。步驟詳見第 7 章與第 8 章。

### 2.2 目錄結構

以下為本專案的完整目錄樹，每個檔案與目錄的用途均有詳細註解：

```
L6-new-model/
├── README.md                          # 專案說明文件（Badge、Quick Start、結構圖）
├── WHITEPAPER.md                      # 技術規格白皮書（本文件，含完整方法論與結果）
├── design.md                          # 架構設計文件（單一事實來源，定義所有模組的職責）
├── requirements.txt                   # Python 相依套件（含版本約束）
├── app.py                             # Streamlit 互動式 CRISP-DM 儀表板（4 章節）
├── main_analysis.py                   # Stage 1-4 完整管線（從資料載入到模型評估）
├── feature_selection.py               # 5 種特徵選取方法比較與投票機制
├── refined_models.py                  # 離群值移除 + Box-Cox + Huber 穩健迴歸
├── outcome_visualization.py           # 循序特徵新增圖表與 CSV/TSV 輸出
├── method_comparison_charts.py        # 5 方法比較儀表板（水平長條圖 + 熱力圖）
├── supplement_analysis.py             # 補充診斷：VIF、5-fold CV、Cook's D、斜率一致性
├── index.html                         # GitHub Pages 靜態儀表板（手繪 Excalidraw 風格）
├── infographic.html                   # 手繪風資訊圖表（與 index.html 同步，含 Interactive / Static Poster 雙頁籤）
├── ml_pipeline_infographic.png        # 靜態高解析度海報（PNG）
├── data/
│   ├── raw/                           # [唯讀] 原始資料 — 絕對不可被程式修改或覆寫
│   │   └── 50_startups.csv           # 50 間新創公司數據（5 欄位）
│   └── processed/                     # [唯寫] 前處理後的訓練/測試集（由管線自動生成）
│       ├── X_train.csv                # 訓練集特徵矩陣（40 筆 × 5 欄）
│       ├── X_test.csv                 # 測試集特徵矩陣（10 筆 × 5 欄）
│       ├── y_train.csv                # 訓練集目標向量（40 筆）
│       └── y_test.csv                 # 測試集目標向量（10 筆）
├── outputs/
│   ├── figures/                       # 11 張 PNG 圖表（自動生成，解析度 150 dpi）
│   │   ├── corr_heatmap.png          # Pearson 相關性熱力圖
│   │   ├── scatter_features.png      # 特徵 vs Profit 散佈圖
│   │   ├── boxplot_state.png         # 不同州 Profit 箱型圖
│   │   ├── pred_vs_actual.png        # 預測值 vs 實際值散佈圖
│   │   ├── residuals.png             # 殘差圖（散佈 + 直方圖）
│   │   ├── feature_count_performance.png  # 特徵數 vs RMSE/R² 雙線圖
│   │   ├── feature_selection_comparison.png # 5 方法特徵選取比較圖
│   │   ├── method_comparison_rmse_r2.png    # 5 方法 RMSE/R² 水平長條圖
│   │   ├── method_comparison_heatmap_performance.png # 熱力圖 + 散佈圖
│   │   ├── method_comparison_dashboard.png  # 4 合 1 全覽儀表板
│   │   └── model_comparison.png      # 精煉模型性能比較圖
│   └── reports/                       # 6 份文字/CSV 報告（自動生成）
│       ├── metrics.txt               # 基礎 OLS 模型完整摘要（statsmodels summary）
│       ├── refined_models.txt        # 精煉模型（A/B/C）比較報告
│       ├── feature_selection.txt     # 5 方法特徵選取文字報告
│       ├── feature_selection_results.csv  # 循序特徵新增數據
│       ├── feature_selection_results.tsv  # 循序特徵新增數據（TSV，Excel 相容）
│       └── method_comparison_results.csv  # 5 方法比較數據
├── .github/workflows/
│   └── deploy.yml                     # GitHub Pages CI/CD 部署管線
├── .gitignore                         # Git 忽略規則（含 .env、__pycache__、venv）
└── .opencode/skills/
    ├── ml-regression-pipeline/         # ML 管線 Skill（可複用 Agent 知識庫）
    │   └── SKILL.md                   # Skill 定義檔（16 步驟完整工作流程）
    └── skill-creator/                  # Skill 建立工具與評估框架
        ├── SKILL.md
        ├── agents/
        ├── scripts/
        ├── eval-viewer/
        └── assets/
```

### 2.3 設計原則

本專案遵循以下五項設計原則，確保程式碼的品質、可維護性與可重現性：

| 原則 | 描述 |
|------|------|
| **唯讀資料層** | `data/raw/` 目錄中的任何檔案不得被任何程式碼修改或覆寫。所有資料轉換結果寫入 `data/processed/`。這確保原始資料的完整性，並允許隨時從源頭重新執行管線。 |
| **集中式前處理** | 所有資料轉換操作（One-Hot Encoding、標準化、Train-Test Split）集中在 `main_analysis.py` 的單一區塊中執行，而非散落在多個檔案。這確保任何對前處理邏輯的修改只需在一個地方進行。 |
| **模組分離** | 每個 `.py` 檔案僅負責單一職責（Single Responsibility Principle）。例如，`feature_selection.py` 僅處理特徵選取邏輯，`refined_models.py` 僅處理模型精煉策略。 |
| **可重現性** | 所有涉及隨機性的操作均使用固定的 `random_state=0`（訓練/測試分割、交叉驗證折疊）。這確保在任何環境中執行相同的程式碼都會得到完全相同的結果。 |
| **Skill 封裝** | 完整管線封裝為可複用的 OpenCode Agent Skill，使得 AI Agent 能在新專案中自動載入此 Skill 並按照相同的方法論執行。Skill 包含架構設計、程式碼範本、診斷檢查表、部署腳本等完整資源。 |

### 2.4 資料流架構

從原始資料到最終部署，資料在系統中的流動路徑如下：

```
[data/raw/50_startups.csv] ──(讀取)──▶ [df_raw DataFrame]
       │
       ▼ (Stage 1: Data Display)
[相關性分析與視覺化] ──▶ [outputs/figures/]
       │
       ▼ (Stage 2: Data Preparation)
[One-Hot Encoding] ──▶ [StandardScaler] ──▶ [Train/Test Split]
       │                                            │
       ▼                                            ▼
[data/processed/X_train.csv, y_train.csv]    [data/processed/X_test.csv, y_test.csv]
       │
       ▼ (Stage 3: Feature Selection + Training)
[5 種特徵選取方法] ──▶ [投票機制] ──▶ [最終特徵組合] ──▶ [OLS 模型訓練]
       │
       ▼ (Stage 4: Evaluation + Refinement)
[模型評估指標] ──▶ [殘差診斷] ──▶ [精煉模型] ──▶ [最終報告]
       │
       ▼ (Deployment)
[GitHub Pages 靜態儀表板] + [Streamlit Cloud 互動式儀表板] + [OpenCode Agent Skill]
```

---

## 3. 資料規格

### 3.1 資料來源

| 屬性 | 值 |
|------|-----|
| 檔案名稱 | `50_startups.csv` |
| 樣本數 | 50 筆 |
| 特徵數 | 4（3 數值 + 1 類別） |
| 目標變數 | Profit（利潤，連續數值） |
| 來源 | Kaggle / 公開學術資料集 |
| 授權 | 公開領域（Public Domain） |
| 格式 | CSV（逗號分隔，UTF-8 編碼） |

### 3.2 欄位結構

| 欄位名稱 | 資料型態 | 描述 | 範圍 | 單位 |
|----------|---------|------|------|------|
| R&D Spend | float64 | 研發支出 | $0 – $165,349.20 | 美元 (USD) |
| Administration | float64 | 行政支出 | $51,283.14 – $182,645.56 | 美元 (USD) |
| Marketing Spend | float64 | 行銷支出 | $0 – $471,784.10 | 美元 (USD) |
| State | object | 公司所在州 | California / Florida / New York | — |
| **Profit** | float64 | **利潤（目標變數）** | $14,681.40 – $192,261.83 | 美元 (USD) |

### 3.3 資料品質檢查

在進行任何分析之前，必須先檢查資料的完整性與品質。以下是本資料集的品質檢查結果：

| 檢查項目 | 方法 | 結果 |
|----------|------|------|
| 缺失值 | `df.isnull().sum()` | **0 個** — 所有 50 筆記錄在 5 個欄位中均無缺失值。 |
| 重複列 | `df.duplicated().sum()` | **0 筆** — 沒有任何完全重複的觀測值。 |
| R&D Spend = 0 | `(df['R&D Spend'] == 0).sum()` | **2 筆** — 兩家公司在該年度完全沒有研發支出。這些是潛在的異常值，但不應在未經分析前直接刪除。 |
| Marketing Spend = 0 | `(df['Marketing Spend'] == 0).sum()` | **3 筆** — 三家公司沒有行銷支出。值得注意的是，其中一筆同時具有 R&D = 0 且 Marketing = 0。 |
| State 類別均衡性 | `df['State'].value_counts()` | California: 17, New York: 17, Florida: 16 — 三個類別近乎均勻分佈，不存在嚴重的類別不平衡問題。 |

### 3.4 數值統計摘要

以下為四個數值欄位的描述性統計（四捨五入至整數位）：

| 欄位 | Mean | Std | Min | 25% | 50% | 75% | Max |
|------|------|-----|-----|------|------|------|------|
| R&D Spend | 73,722 | 45,902 | 0 | 39,936 | 73,051 | 101,603 | 165,349 |
| Administration | 121,345 | 28,018 | 51,283 | 103,731 | 122,700 | 144,842 | 182,646 |
| Marketing Spend | 211,025 | 122,290 | 0 | 129,300 | 212,716 | 299,469 | 471,784 |
| Profit | 112,013 | 40,306 | 14,681 | 90,139 | 107,978 | 139,766 | 192,262 |

從上表可以觀察到幾個重要特徵：

- **R&D Spend** 的分佈範圍極廣（從 $0 到 $165,349），標準差接近平均值的 62%，顯示不同新創公司在研發投入上有極大差異。這意味著 R&D 支出具有很高的變異性，可能是解釋利潤差異的關鍵變數。
- **Administration** 的變異係數（Coefficient of Variation, CV = Std/Mean）約為 23%，相對穩定。大多數公司的行政支出集中在 $100,000–$145,000 之間。
- **Marketing Spend** 的標準差（$122,290）超過平均值的一半，且有 3 筆為零支出。行銷支出的分佈呈現高度右偏（right-skewed），少數公司投入巨額行銷預算。
- **Profit** 的分佈從 $14,681 到 $192,262，跨度超過 13 倍。這表示資料集中同時包含了小型與大型新創公司，模型需要能夠準確預測整個範圍內的利潤值。

### 3.5 State 類別分佈

類別變數 State 的頻率分佈如下：

| State | 樣本數 | 百分比 |
|-------|:---:|:---:|
| California | 17 | 34.0% |
| New York | 17 | 34.0% |
| Florida | 16 | 32.0% |

三個州別的樣本數幾乎相等，避免了類別不平衡可能導致的估計偏差。各州利潤的基本統計如下：

| State | Mean Profit | Median Profit | Std Profit |
|-------|:---:|:---:|:---:|
| California | $109,508 | $97,428 | $46,808 |
| New York | $108,460 | $108,552 | $41,989 |
| Florida | $118,296 | $124,267 | $33,152 |

佛羅里達州（Florida）的平均利潤略高於其他兩州，但標準差也較大，暗示州別對利潤的影響可能有限。

### 3.6 探索性資料分析 (EDA) 指南

在執行建模之前，進行完整的探索性資料分析可以幫助理解資料的基本結構與潛在問題。以下是本專案執行的 EDA 步驟及其目的：

**步驟 1：單變量分析 (Univariate Analysis)**
- 使用 `df.describe()` 獲得數值變數的集中趨勢（mean, median）與離散程度（std, min, max, quartiles）。
- 使用 `df['State'].value_counts()` 檢查類別變數的頻率分佈。
- 目的：識別異常的極端值、檢查量綱（scale）是否需要標準化。

**步驟 2：雙變量分析 (Bivariate Analysis)**
- 使用 `df.corr()` 計算 Pearson 相關係數矩陣。Pearson r 衡量兩個連續變數之間的線性相關程度，取值範圍 [-1, +1]。
- 繪製散佈圖（scatter plot）觀察每個特徵與目標變數之間的關係模式。特別注意是否存在非線性關係（如 U 型或指數型），這可能暗示需要進行資料轉換。
- 使用箱型圖（box plot）比較不同類別（State）下目標變數的分佈差異。

**步驟 3：相關性解讀**
- R&D Spend 與 Profit 的 Pearson r ≈ 0.973，顯示極強的線性正相關。這暗示 R&D 支出是利潤最強的預測因子，單一變數可能就能解釋大部分利潤的變異。
- Marketing Spend 與 Profit 的 r ≈ 0.748，顯示中高程度的正相關，但強度明顯不如 R&D。
- Administration 與 Profit 的 r ≈ 0.201，相關性相當微弱。這是一個重要的早期信號——行政支出可能不是利潤的有效預測因子。

### 3.7 相關係數的深入解讀：區分統計顯著性與實務意義

在進行相關性分析時，一個常見的陷阱是過度解讀相關係數的大小而忽略了其統計顯著性和實務意義。以下是一個有助於決策的框架：

**統計顯著性 vs. 效果大小**：
- 統計顯著性（P-value）告訴我們「這個相關性是否可能只是偶然？」
- 效果大小（相關係數 r 本身）告訴我們「這個相關性在實務上有多重要？」

在 50 筆資料中，要達到 P < 0.05 的統計顯著性，所需的相關係數閾值約為 |r| > 0.279。因此：
- R&D Spend（r = 0.973）：同時具有統計顯著性和極大的效果大小
- Marketing Spend（r = 0.748）：同時具有統計顯著性和中大的效果大小
- Administration（r = 0.201）：**不具統計顯著性**，效果大小也接近可忽略

**相關係數的限制**：
Pearson 相關係數僅衡量**線性**關係的強度。如果一個特徵與目標之間存在 U 型關係（如中等支出產生最高利潤，而低支出和高支出都產生較低利潤），Pearson r 可能接近於零，給人「無關聯」的錯誤印象。雖然在本資料集中未發現強烈的非線性關係，但在其他應用場景中應考慮使用 Spearman 秩相關係數（捕捉單調關係）或 Mutual Information（捕捉任何統計依賴性）作為補充。

**為何不移除低相關的特徵？**：
一個常見的初學者錯誤是僅根據相關性分析就移除「看似不相關」的特徵。然而：
1. 相關性分析是**單變量**的（一次只看一個特徵），無法捕捉特徵組合的協同效應
2. 一個與目標弱相關的特徵可能在與其他特徵結合後變得重要（交互作用）
3. 在某些情況下，低相關特徵可能是「抑制變數」（suppressor variable），其存在能提高其他特徵的預測準確度

因此，相關性分析應僅用於初步探索，不應作為特徵選取的最終依據。本專案使用五種方法（包括多變量方法如 Backward Elimination 和 RFE）來做最終的特徵選取決策。

### 3.8 資料視覺化的最佳實踐

以下是本專案中使用的視覺化策略及其設計理由，可供讀者在自己的 EDA 中參考：

| 視覺化類型 | 適用場景 | 設計要點 |
|-----------|---------|---------|
| **熱力圖（Heatmap）** | 展示多個變數之間的相關性矩陣 | 使用發散型色彩映射（如 RdBu），以 0 為中心對稱著色；加上數值標註以提供精確數據；使用上三角遮罩避免資訊重複 |
| **散佈圖（Scatter Plot）** | 觀察兩個連續變數之間的關係 | 加入透明度（alpha）以顯示重疊點的密度；在標題中標註相關係數值 |
| **箱型圖（Box Plot）** | 比較不同類別下的數值分佈 | 按中位數排序類別以利視覺比較；可疊加蜂群圖（stripplot）顯示個別資料點 |
| **殘差圖（Residual Plot）** | 診斷迴歸模型假設 | 加入 y = 0 的參考線；使用直方圖輔助檢查殘差的常態性 |

---

## 4. 資料前處理管線

### 4.1 處理步驟（嚴格依序）

資料前處理是機器學習專案中最關鍵也最耗時的階段。一個設計不良的前處理管線可能導致資料洩漏（Data Leakage）或錯誤的模型評估。本專案的前處理管線包含七個步驟，必須嚴格依序執行：

```
Step 1: 複製原始資料 ──▶ df = df_raw.copy()
Step 2: One-Hot Encoding ──▶ pd.get_dummies(df["State"], prefix="State", dtype=int)
Step 3: 避開虛擬變數陷阱 ──▶ 刪除第一個 dummy 欄位（基準類別：California）
Step 4: 分離特徵與目標 ──▶ X = df.drop(columns=["Profit"]), y = df["Profit"]
Step 5: 標準化數值特徵 ──▶ StandardScaler（僅套用於 R&D, Admin, Marketing）
Step 6: 訓練/測試分割 ──▶ train_test_split(test_size=0.2, random_state=0)
Step 7: 保存處理後資料 ──▶ CSV 輸出至 data/processed/
```

### 4.2 Step 1：複製原始資料

在進行任何轉換之前，先使用 `.copy()` 創建原始 DataFrame 的深層複本（deep copy）。這確保後續的所有操作不會意外修改到載入的原始資料：

```python
df = df_raw.copy()  # 創建獨立副本
```

深層複本保證了 `df` 和 `df_raw` 在記憶體中是兩個完全獨立的物件，修改其中一個不會影響另一個。

### 4.3 Step 2：One-Hot Encoding

類別變數（State）無法直接輸入線性迴歸模型，因為數學運算（如矩陣乘法）需要數值輸入。One-Hot Encoding 是處理類別變數最常用的方法：將每個類別轉換為一個二元（0/1）虛擬變數。

```python
state_dummies = pd.get_dummies(df["State"], prefix="State", dtype=int)
# 輸出：State_California, State_Florida, State_New York（三個欄位，每列只有一個 1）
```

`dtype=int` 確保虛擬變數以整數 0/1 儲存（而非布林值），這在後續的矩陣運算中更為高效。

### 4.4 Step 3：避開虛擬變數陷阱

虛擬變數陷阱（Dummy Variable Trap）是多元線性迴歸中的一個經典問題。當我們保留了 k 個虛擬變數（對應 k 個類別）時，這 k 個變數存在完全共線性——任何一個變數都等於 1 減去其餘變數之和。這會導致正規方程式 (XᵀX)⁻¹ 不可逆，使 OLS 估計失效。

解決方案是刪除其中一個虛擬變數，將其作為「基準類別」（Baseline Category）。剩餘的 k-1 個變數完全足以表達所有 k 個類別的資訊，同時避免了共線性問題。

```python
drop_col = state_dummies.columns[0]       # "State_California"
state_dummies = state_dummies.drop(columns=[drop_col])
# 保留：State_Florida, State_New York
# 基準：California（當兩者皆為 0 時隱含表示 California）
```

| State 原始值 | State_Florida | State_New York | 隱含意義 |
|-------------|:---:|:---:|------|
| California | 0 | 0 | 基準類別 |
| Florida | 1 | 0 | — |
| New York | 0 | 1 | — |

### 4.5 Step 5：標準化數值特徵

線性迴歸對特徵的量綱（scale）不敏感，但標準化在以下情境中非常重要：

- 正則化方法（如 Lasso、Ridge）要求特徵具有相似的量綱，否則懲罰項會不公平地壓縮量綱較大的係數。
- 基於係數大小的特徵選取方法（如 RFE）在未標準化的資料上可能產生誤導。
- 梯度下降優化在標準化後的資料上收斂更快（雖然 OLS 使用封閉解，不涉及迭代）。

本專案使用 Z-score 標準化（StandardScaler），將每個數值特徵轉換為平均值 0、標準差 1 的分佈：

```
z = (x - μ) / σ
```

其中 μ 為特徵在訓練集中的平均值，σ 為標準差。

**關鍵原則**：僅使用訓練集來擬合（fit）標準化器，然後使用同一個標準化器來轉換（transform）測試集。這避免了資料洩漏——測試集的資訊不應影響任何前處理參數的估計。

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[num_features] = scaler.fit_transform(X[num_features])
# fit_transform 在訓練時同時擬合並轉換
# 對測試集：僅使用 .transform()，不重新擬合
```

**標準化範圍**：僅對三個數值特徵（R&D Spend、Administration、Marketing Spend）進行標準化。虛擬變數（State_Florida、State_New York）已經是 0/1 二元值，不需要也不應該被標準化。

### 4.6 Step 6：訓練/測試分割

將資料集劃分為訓練集（用於模型訓練）和測試集（用於效能評估）是監督式學習的標準做法。本專案使用 80/20 分割比例：

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=0
)
```

- **訓練集**：40 筆（80%），用於擬合模型參數和進行特徵選取。
- **測試集**：10 筆（20%），用於評估最終模型的泛化能力。測試集在整個模型開發過程中保持「未被觸碰」的狀態，僅在最終評估時使用一次。
- **random_state=0**：固定隨機種子，確保每次執行時獲得完全相同的分割結果。這對可重現性至關重要。

### 4.7 常見陷阱與最佳實踐

以下是資料前處理中幾個容易犯的錯誤及其避免方法：

| 陷阱 | 錯誤做法 | 正確做法 |
|------|---------|---------|
| **資料洩漏** | 在分割前對整個資料集進行標準化 | 先分割，再用訓練集的 mean/std 標準化測試集 |
| **虛擬變數陷阱** | 保留所有 k 個虛擬變數 | 刪除一個作為基準類別（k-1 編碼） |
| **不當標準化** | 對二元虛擬變數進行 Z-score 標準化 | 僅標準化連續數值特徵 |
| **修改原始資料** | 直接對 `df_raw` 進行操作 | 使用 `.copy()` 創建副本後再操作 |
| **忽略隨機種子** | 不使用 `random_state` 或每次使用不同值 | 固定 `random_state=0` 確保可重現性 |

---

## 5. 特徵選取方法論

特徵選取（Feature Selection）是機器學習管線中的關鍵步驟，目的是從可用特徵中識別出對預測目標最有貢獻的子集。適當的特徵選取可以帶來以下好處：

1. **降低過擬合風險**：減少不必要的特徵可以降低模型捕捉訓練資料中隨機雜訊的機會。
2. **提高模型可解釋性**：較少的特徵意味著更簡單的模型，更容易向非技術利益相關者解釋。
3. **減少訓練與推論時間**：特徵數量直接影響模型訓練和預測的計算成本。
4. **避免多重共線性**：移除高度相關的多餘特徵有助於穩定係數估計。

### 5.1 五種方法總覽

本專案同時採用五種特徵選取方法，涵蓋特徵選取的三個主要流派：統計檢定（Filter 方法）、包裝法（Wrapper 方法）、嵌入式（Embedded 方法）。透過多方法交叉驗證，我們可以對特徵重要性得出更穩健的結論。

| # | 方法 | 類別 | 核心機制 | 超參數 | 輸出 |
|---|------|------|---------|--------|------|
| 1 | Backward Elimination | Filter（統計檢定） | 從全模型開始，每輪剔除 P-value 最高的不顯著變數 | α = 0.05 | 保留特徵清單 |
| 2 | Forward Selection | Filter（統計檢定） | 從空模型開始，每輪加入 P-value 最低的顯著變數 | α = 0.05 | 保留特徵清單 |
| 3 | RFE（Recursive Feature Elimination） | Wrapper（包裝法） | 迭代訓練線性模型，每輪淘汰 |coef| 最小的特徵 | 5-fold CV 選最優特徵數 | 最優特徵數 + 特徵清單 |
| 4 | Lasso Regression（L1 正則化） | Embedded（嵌入式） | L1 懲罰項將不重要特徵的係數自動壓縮至 0 | 5-fold CV 選 α | 係數非零的特徵清單 |
| 5 | Mutual Information（互信息） | Filter（過濾法） | 衡量每個特徵與目標變數之間的資訊共享量 | 閾值 = max(MI) × 0.2 | MI 值高於閾值的特徵清單 |

### 5.2 方法 1：Backward Elimination（逐步淘汰法）

**原理**：Backward Elimination 是一種基於 P-value 的逐步特徵選取方法。它從包含所有特徵的「全模型」開始，在每一輪迭代中識別 P-value 最高的特徵（即統計上最不顯著的特徵）。如果該 P-value 超過預設的顯著水準 α（預設 0.05），則將該特徵從模型中移除。這個過程持續進行，直到模型中所有剩餘特徵的 P-value 都小於或等於 α。

**P-value 的直觀解釋**：P-value 回答了以下問題：「如果該特徵實際上對目標沒有任何影響（虛無假設為真），我們觀察到當前（或更極端）的係數估計值的機率是多少？」較小的 P-value 意味著該特徵不太可能是偶然產生的效果，更有可能是真正的預測因子。

**演算法流程**：
```
1. 使用所有特徵擬合 OLS 模型
2. 計算每個特徵的 P-value
3. 找出 P-value 最大的特徵
4. 如果 max(P-value) > α (0.05)：
     移除該特徵，回到步驟 1
   否則：
     停止，保留當前模型中的所有特徵
```

**Python 實作**：使用 `statsmodels.OLS` 進行模型擬合，因為 scikit-learn 的 `LinearRegression` 不提供 P-value。`statsmodels` 提供了完整的推論統計，包括係數估計值、標準誤、t 統計量、P-value、信賴區間等。

```python
import statsmodels.api as sm

def backward_elimination(X, y, sl=0.05):
    features = list(X.columns)
    while True:
        X_sm = sm.add_constant(X[features])
        model = sm.OLS(y, X_sm).fit()
        pvalues = model.pvalues.drop('const')
        max_p = pvalues.max()
        if max_p > sl:
            worst_feature = pvalues.idxmax()
            features.remove(worst_feature)
        else:
            break
    return features
```

**優點**：直觀易懂，P-value 是統計學中廣為人知的概念，結果易於解釋給非技術背景的利害關係人。同時考慮了特徵之間的交互作用（因為每次都是重新擬合整個模型）。

**限制**：P-value 受樣本大小影響（大樣本時即使小的效果也可能顯著）。逐步程序可能收斂到局部最優而非全局最優（取決於特徵的移除順序）。此外，P-value 假設殘差服從常態分佈，當此假設不成立時結果可能不可靠。

### 5.3 方法 2：Forward Selection（逐步選擇法）

**原理**：Forward Selection 是 Backward Elimination 的鏡像方法。它從一個空模型（僅含截距項）開始，在每一輪迭代中，對於尚未被選入的每個特徵，分別擬合一個包含該特徵的模型，並選取 P-value 最低的特徵。如果該最低 P-value 小於 α（0.05），則將該特徵加入模型。這個過程持續進行，直到沒有任何未選取特徵的 P-value 低於 α。

**演算法流程**：
```
1. 從空特徵集開始
2. 對於每個尚未選入的特徵：
     擬合「已選特徵 + 該特徵」的 OLS 模型
     記錄該特徵的 P-value
3. 找出 P-value 最小的候選特徵
4. 如果 min(P-value) < α (0.05)：
     將該特徵加入已選集，回到步驟 2
   否則：
     停止，返回已選特徵集
```

**Backward vs. Forward 的選擇**：當特徵數量較多時，Forward Selection 通常更高效（因為從空模型開始，早期迭代的計算量較小）。Backward Elimination 則能更好地捕捉特徵之間的協同效應（因為一開始就考慮了所有特徵）。在實務中，兩者經常被同時執行以交叉驗證結果。

### 5.4 方法 3：Recursive Feature Elimination（RFE）

**原理**：RFE 是一種包裝法（Wrapper Method），它與一個基礎估計器（此處為 `LinearRegression`）配合使用。RFE 迭代地訓練模型，並在每輪淘汰係數絕對值（|coef|）最小的特徵。這個「訓練→淘汰」的循環重複進行，直到達到指定的特徵數量。

**為什麼淘汰 |coef| 最小的特徵？** 在標準化後的資料上，線性迴歸係數的大小反映了該特徵對預測值的邊際貢獻。絕對值越小的係數意味著該特徵在模型中「權重」越低，對預測的影響越小。

**最優特徵數選擇**：RFE 本身需要指定最終要保留多少個特徵。為了客觀地決定這個數量，我們遍歷 n = 1 到所有特徵數，對每個數量使用 RFE 選取特徵後以 5-fold 交叉驗證評估 R²，選擇能最大化 CV R² 的特徵數量：

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

best_n = 1
best_score = -float('inf')
for n in range(1, X.shape[1] + 1):
    rfe = RFE(LinearRegression(), n_features_to_select=n)
    X_rfe = rfe.fit_transform(X_train, y_train)
    scores = cross_val_score(LinearRegression(), X_rfe, y_train, cv=5)
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_n = n
```

**優點**：RFE 考慮了特徵之間的相互關係（因為每次淘汰後會重新訓練），相較於單純的單變量過濾法更為穩健。

**限制**：計算成本較高（對於大量特徵需要多次迭代）。對基礎估計器的選擇敏感。

### 5.5 方法 4：Lasso Regression（L1 正則化）

**原理**：Lasso（Least Absolute Shrinkage and Selection Operator）是一種嵌入式特徵選取方法。它在標準的 OLS 損失函數上增加了一個 L1 懲罰項，使得模型在最小化預測誤差的同時，也傾向於將部分係數「壓縮」至恰好為零。

**損失函數**：
```
L(β) = Σ(yᵢ - Xᵢβ)² + α × Σ|βⱼ|
       └── RSS ──┘   └─ L1 Penalty ─┘
```

其中 α（alpha）是正則化強度超參數：
- α → 0：Lasso 趨近於普通 OLS（無懲罰）
- α → ∞：所有係數被壓縮至 0（空模型）

**為什麼 L1 會產生稀疏解？** L1 懲罰項的等高線在座標軸上形成「尖角」，當損失函數的等高線與這些尖角相交時，最優解恰好落在座標軸上（某些係數恰為 0）。相較之下，L2（Ridge）懲罰項的等高線是圓形，最優解通常不在座標軸上，因此不會產生精確的零係數。

**α 的選擇**：使用 `LassoCV` 以 5-fold 交叉驗證自動選擇最優的 α。預設搜尋範圍為 α ∈ [10⁻⁴, 10²] 中的 100 個對數均勻間隔的值：

```python
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5, max_iter=10000, random_state=0)
lasso.fit(X_train, y_train)

# 選取係數非零的特徵
selected = X.columns[np.abs(lasso.coef_) > 1e-5].tolist()
```

**優點**：Lasso 同時執行了特徵選取（係數歸零）和正則化（防止過擬合），是一種高效的一站式解決方案。

**限制**：當特徵之間存在高度相關（多重共線性）時，Lasso 傾向於隨機選取其中一個而將其他歸零，結果可能不穩定。在這種情況下，Elastic Net（結合 L1 + L2）可能是更好的選擇。

### 5.6 方法 5：Mutual Information（互信息）

**原理**：Mutual Information（MI）源自資訊理論，衡量兩個隨機變數之間的相互依賴程度。與 Pearson 相關係數僅捕捉線性關係不同，MI 能夠檢測任何形式的統計依賴性（線性或非線性）。

**數學定義**：
```
I(X; Y) = Σₓ Σᵧ p(x,y) × log( p(x,y) / (p(x) × p(y)) )
```

直觀解釋：MI 量化了「知道 X 的值能減少多少關於 Y 的不確定性」。如果 X 和 Y 完全獨立，則 MI = 0；如果 X 完全決定了 Y，則 MI 達到最大值。

**閾值策略**：計算每個特徵與目標變數之間的 MI 值後，設定一個動態閾值來決定保留哪些特徵。本專案使用「最大 MI 值的 20%」作為閾值——這確保了只保留與目標有實質性關聯的特徵，同時自動適應不同資料集的 MI 絕對值範圍。

```python
from sklearn.feature_selection import mutual_info_regression

mi_scores = mutual_info_regression(X_train, y_train, random_state=0)
threshold = mi_scores.max() * 0.2
selected = X.columns[mi_scores >= threshold].tolist()
```

**優點**：能夠檢測非線性關係，不需要預先假設模型形式。對特徵的量綱不敏感。

**限制**：MI 是一個單變量指標（一次只考慮一個特徵與目標的關係），不考量特徵之間的交互作用。此外，MI 的估計在連續變數上需要離散化或核密度估計，小樣本時估計可能不穩定。

### 5.7 投票機制與共識建立

由於不同的特徵選取方法基於不同的假設與機制（統計顯著性、模型係數、正則化、資訊理論），它們的結果可能不完全一致。為了在這些方法之間建立共識，本專案採用了一個簡單而透明的投票機制：

```
特徵得票數 = Σ（各方法是否選取該特徵）

其中「是否選取」為二元值：選取 = 1, 未選取 = 0
最大可能得票數 = 5（五種方法全部選取）
```

**共識層級**：
- **全票通過（5/5）**：所有五種方法一致認可的特徵。這些特徵具有無可爭議的預測價值，應始終包含在模型中。
- **多數共識（≥3/5）**：獲得多數方法支持的特徵。這些特徵具有較高的可信度，但可能需要根據具體應用場景權衡是否包含。
- **弱證據（2/5）**：僅有少數方法支持的特徵。其預測價值存疑，建議進行更深入的敏感性分析。
- **噪音（≤1/5）**：幾乎被所有方法否決的特徵，應從模型中排除。

這個投票機制不僅提供了特徵選取的最終決策，還為每個特徵建立了「信心分數」，有助於利害關係人理解模型選擇的背後邏輯。

### 5.8 特徵選取實戰演練：逐步追蹤分析過程

為了讓讀者完全理解特徵選取是如何在實務中進行的，以下逐輪追蹤 Backward Elimination 在 50 Startups 資料集上的完整執行過程。初始模型包含全部 5 個特徵（R&D Spend、Administration、Marketing Spend、State_Florida、State_New York）。

**第 1 輪：全模型評估**

首先使用所有 5 個特徵擬合 OLS 模型。statsmodels 的輸出包含每個係數的估計值、標準誤、t 統計量、P-value，以及 95% 信賴區間。以下是第 1 輪的關鍵輸出：

| 特徵 | 係數 | 標準誤 | t 值 | P-value | 判定 |
|------|------|--------|------|---------|------|
| const (截距) | 111,199 | 2,522 | 44.09 | 0.000 | — |
| R&D Spend | 38,699 | 3,688 | 10.49 | 0.000 | 極顯著 |
| Administration | −861 | 2,479 | −0.35 | 0.730 | 不顯著 |
| Marketing Spend | 3,339 | 3,738 | 0.89 | 0.377 | 不顯著 |
| State_Florida | 1,351 | 3,910 | 0.35 | 0.733 | 不顯著 |
| State_New York | 96 | 3,936 | 0.02 | 0.981 | 極不顯著 |

檢視各特徵的 P-value：
- R&D Spend 的 P-value = 0.000（極小），確定是顯著的預測因子
- State_New York 的 P-value = 0.981 是最高的，且遠大於 α = 0.05

因此，第 1 輪淘汰 State_New York。這符合我們的直覺——紐約州的虛擬變數對利潤幾乎沒有任何解釋力。

**第 2 輪：移除 State_New York 後**

重新擬合包含 4 個特徵的模型：

| 特徵 | 係數 | P-value |
|------|------|---------|
| R&D Spend | 38,677 | 0.000 |
| Administration | −855 | 0.724 |
| Marketing Spend | 3,341 | 0.368 |
| State_Florida | 1,306 | 0.657 |

Administration 的 P-value = 0.724 是本輪最高的，超過 α = 0.05。淘汰 Administration。

**第 3 輪：移除 Administration 後**

重新擬合包含 3 個特徵的模型：

| 特徵 | 係數 | P-value |
|------|------|---------|
| R&D Spend | 38,011 | 0.000 |
| Marketing Spend | 2,748 | 0.136 |
| State_Florida | 1,480 | 0.605 |

State_Florida 的 P-value = 0.605 是本輪最高的，超過 α = 0.05。淘汰 State_Florida。

**第 4 輪：移除 State_Florida 後**

重新擬合包含 2 個特徵的模型：

| 特徵 | 係數 | P-value |
|------|------|---------|
| R&D Spend | 37,893 | 0.000 |
| Marketing Spend | 2,894 | 0.112 |

Marketing Spend 的 P-value = 0.112，雖然較之前有所改善，但仍然超過 α = 0.05。淘汰 Marketing Spend。

**第 5 輪：僅剩 R&D Spend**

僅包含 R&D Spend 的模型（加上截距項）：

| 特徵 | 係數 | P-value |
|------|------|---------|
| R&D Spend | 39,829 | 0.000 |

所有剩餘特徵的 P-value 均小於或等於 0.05。Backward Elimination 終止。

**最終結果**：Backward Elimination 僅保留了 R&D Spend 一個特徵。這是 P-value 方法的嚴格性在運作——在 α = 0.05 的標準下，Marketing Spend 的邊際貢獻不足以達到統計顯著性。

**為什麼 Marketing 在 α = 0.05 下不顯著？** 這背後有兩個主要原因：
1. R&D 與 Marketing 之間存在中等程度的相關性（約 0.72），這意味著 R&D 已經在一定程度上「代表」了 Marketing 的預測資訊。當 R&D 已在模型中時，Marketing 能提供的「額外」解釋力有限。
2. 樣本量小（n = 40 個訓練樣本），使得即使有真實效果的特徵也可能無法達到統計顯著性（統計檢定力不足）。

**實務意義**：這正是為什麼我們不只依賴單一方法的原因。Mutual Information 和 RFE 都認為 Marketing 具有預測價值，且循序特徵新增分析清楚地顯示加入 Marketing 能降低 RMSE。在商業應用中，如果模型的用途是預測（而非推論因果關係），保留 Marketing 是有益的——即使它在嚴格的統計檢定下不顯著。

### 5.9 方法選擇的實務建議

在不同的商業場景中，應根據目標選擇不同的特徵選取策略：

| 場景 | 推薦方法 | 理由 |
|------|---------|------|
| **學術研究 / 因果推論** | Backward / Forward Selection | P-value 提供了嚴格的統計證據，適合發表與同儕審查 |
| **預測競賽 / 最大化 R²** | Mutual Information | 能捕捉非線性關係，通常能找到預測最優的特徵組合 |
| **高維度資料（p >> n）** | Lasso L1 | L1 正則化在特徵數超過樣本數時仍能有效運作 |
| **需要穩健模型** | RFE + CV | 交叉驗證降低了過擬合風險，RFE 使用係數權重而非 P-value |
| **混合策略（推薦）** | 投票機制（5 種方法） | 綜合多種方法的優點，避免單一方法的盲點 |

在本專案中，我們採用混合策略，並發現 R&D Spend（5/5 票）和 Marketing Spend（3/5 票）是獲得跨方法共識的最佳特徵組合。

---

## 6. 模型實作細節

### 6.1 基礎模型：Ordinary Least Squares（OLS）

**數學形式**：多元線性迴歸模型假設目標變數 y 與特徵矩陣 X 之間存在線性關係：

```
y = Xβ + ε
```

其中：
- y ∈ ℝⁿ 是 n 個觀測值的目標向量
- X ∈ ℝⁿˣᵖ 是 n × p 的設計矩陣（含截距項時第一欄全為 1）
- β ∈ ℝᵖ 是 p 個迴歸係數（待估計）
- ε ∈ ℝⁿ 是誤差項，假設 εᵢ ~ N(0, σ²) 且相互獨立

**OLS 估計**：最小平方法選擇能最小化殘差平方和（Residual Sum of Squares, RSS）的 β：

```
β̂ = argmin_β Σ(yᵢ - xᵢβ)² = (XᵀX)⁻¹Xᵀy
```

這個封閉解（closed-form solution）是 OLS 的核心，它保證了在滿足高斯-馬可夫假設（Gauss-Markov Assumptions）的條件下，β̂ 是最佳線性不偏估計量（BLUE, Best Linear Unbiased Estimator）。

**雙重實作**：本專案同時使用兩個函式庫來確保結果的可靠性與互補性：

- **statsmodels.OLS**：提供完整的推論統計，包括係數的標準誤、t 統計量、P-value、信賴區間、以及模型整體的 F 檢定、R²、AIC、BIC。適合模型解釋與報告生成。
- **sklearn.LinearRegression**：提供與 scikit-learn 生態系統的無縫整合（如 Pipeline、GridSearchCV），便於與其他特徵選取和驗證工具配合使用。

### 6.2 精煉模型變體：處理資料缺陷

基礎 OLS 模型假設資料是完美的——沒有異常值、殘差服從常態分佈、變異數齊一。然而，真實世界的資料很少滿足這些理想條件。本專案實作了四種精煉策略來處理不同類型的資料缺陷：

| 變體 | 描述 | 觸發條件 | 改善目標 |
|------|------|---------|---------|
| **Model A: Cleaned OLS** | 移除高影響點（基於 Cook's Distance）後重新訓練 OLS | 任一點 Cook's D > 4/n | 降低異常值對係數估計的干擾 |
| **Model B: Box-Cox** | 對目標變數 y 做 Box-Cox 冪次轉換後建模，再反轉換回原始尺度 | 殘差 Omnibus normality test p < 0.05 | 使殘差更接近常態分佈 |
| **Model C: Huber** | 使用 Huber Loss 替代平方損失，降低離群值的影響權重 | 穩健性驗證需求 | 對離群值不敏感 |
| **Model D: No_RD Flag** | 加入一個二元旗標變數標記「R&D = 0」的樣本 | R&D = 0 的樣本存在 | 區分「無研發」與「有研發」的結構性差異 |

**Model A — Cleaned OLS**：Cook's Distance 衡量每個觀測值對整個迴歸模型的影響力。數值越大，表示移除該點後模型係數會發生顯著變化。常用的閾值為 4/n（其中 n 是樣本數）。移除超過此閾值的點後重新訓練模型，可以獲得更穩健的係數估計。

**Model B — Box-Cox 轉換**：當殘差偏離常態分佈時，可以對目標變數應用 Box-Cox 冪次轉換：

```
y(λ) = { (y^λ - 1) / λ,  if λ ≠ 0
       { ln(y),          if λ = 0
```

λ 參數透過最大概似估計（MLE）自動選擇，目標是使轉換後的變數盡可能接近常態分佈。預測時需要進行反轉換以回到原始尺度。

**Model C — Huber 穩健迴歸**：Huber Loss 結合了 MSE（對小誤差的敏感性）和 MAE（對大誤差的不敏感性）：

```
L_δ(r) = { 0.5 × r²,           if |r| ≤ δ
         { δ × (|r| - 0.5δ),   if |r| > δ
```

其中 δ 是閾值參數。當殘差較小時（|r| ≤ δ），使用平方損失以保持效率；當殘差較大時（|r| > δ），切換為線性損失以降低離群值的影響。

### 6.3 評估指標詳解

模型評估不僅需要單一指標，而是需要一套完整的指標矩陣來從不同角度衡量模型性能：

| 指標 | 公式 | 值域 | 理想值 | 用途 |
|------|------|------|--------|------|
| **R²** | 1 − SS_res / SS_tot | (−∞, 1] | 接近 1 | 解釋變異比例，模型「有多好」的直觀度量 |
| **Adjusted R²** | 1 − (1−R²)(n−1)/(n−p−1) | (−∞, 1] | 接近 1 | 懲罰不必要的特徵數量，防止「加特徵就提升 R²」的錯覺 |
| **MAE** | (1/n) Σ | yᵢ − ŷᵢ | | [0, ∞) | 0 | 平均絕對誤差（與目標相同單位），對離群值不敏感 |
| **RMSE** | √( (1/n) Σ (yᵢ − ŷᵢ)² ) | [0, ∞) | 0 | 均方根誤差，對大誤差施以較重懲罰，適合需要嚴控大偏離的情境 |
| **5-fold CV R²** | 五折交叉驗證 R² 的均值 ± 標準差 | (−∞, 1] | 接近 1，且 Std 小 | 泛化能力評估，標準差反映模型在不同資料子集上的穩定性 |

**R² vs. Adjusted R²**：R² 的一個關鍵缺陷是它永遠不會因增加更多特徵而下降（即使新增的是隨機雜訊）。Adjusted R² 通過引入特徵數量的懲罰來修正這個問題——只有當新增特徵的貢獻超過其「成本」時，Adjusted R² 才會提升。因此，Adjusted R² 是比較不同特徵數量模型時的更可靠指標。

**MAE vs. RMSE**：RMSE 對大誤差的懲罰更重（因為先平方再開根號）。在利潤預測的情境中，這意味著 RMSE 更能反映嚴重錯誤預測的成本——一個讓預測偏離 $20,000 的模型比兩個各偏離 $10,000 的預測更糟糕（RMSE 會捕捉到這一點，MAE 則不會）。

---

## 7. 實驗結果

### 7.1 相關性分析

在進行任何建模之前，先透過相關性分析建立對資料結構的初步理解：

| 特徵 | Pearson r vs Profit | 相關性強度 | 解釋 |
|------|-----|------|------|
| R&D Spend | 0.9729 | 極強正相關 | R&D 支出與利潤幾乎成完美線性關係。這暗示 R&D 可能是單一最強預測因子。 |
| Marketing Spend | 0.7478 | 強正相關 | 行銷支出與利潤有顯著的正向關係，但強度明顯不如 R&D。 |
| Administration | 0.2007 | 弱正相關 | 行政支出與利潤的關係相當微弱，可能不具備預測價值。 |

**行政支出的角色**：Administration 與 Profit 的相關性僅 0.20，這意味著行政支出的變異僅能解釋約 4%（0.20² = 0.04）的利潤變異。從投資角度來看，這暗示削減或增加行政支出可能不會對利潤產生顯著影響——這是一個重要的商業洞察。

### 7.2 五種特徵選取方法比較

以下是五種方法在測試集上的性能比較，按測試 R² 降冪排序：

| 方法 | 選取特徵數 | 選取特徵 | 測試 R² | Adj. R² | 測試 RMSE |
|------|:---:|------|:---:|:---:|---:|
| Mutual Info (top) | 2 | R&D, Marketing | **0.9474** | 0.9324 | $8,198.80 |
| Backward Elimination | 1 | R&D | 0.9465 | **0.9398** | $8,274.87 |
| Forward Selection | 1 | R&D | 0.9465 | **0.9398** | $8,274.87 |
| RFE (CV-optimal) | 3 | R&D, Marketing, State_FL | 0.9451 | 0.9177 | $8,376.45 |
| Lasso L1 | 5 | All features | 0.9347 | 0.8531 | $9,137.99 |

**關鍵觀察**：

1. **Mutual Information 達到最高 R²**（0.9474）和最低 RMSE（$8,198.80），選出 R&D + Marketing 兩個特徵。這個結果與循序特徵新增分析中特徵數 = 2 為最優點的結論完全一致。

2. **Backward 和 Forward 選擇只保留了 R&D**。這反映了 P-value 為基礎的方法的保守性——在 α = 0.05 的嚴格標準下，Marketing Spend 的邊際貢獻可能未達到統計顯著性。

3. **RFE 選擇了 3 個特徵**（加了 State_Florida），但測試集性能（R² = 0.9451）反而略低於兩特徵模型。這是一個經典的過擬合信號——RFE 在訓練集上找到了包含 State_Florida 的模型略優，但這個優勢未能泛化到測試集。

4. **Lasso 保留了所有 5 個特徵**，但性能最差（R² = 0.9347, RMSE = $9,137.99）。值得注意的是 Lasso 的 Adjusted R² 僅 0.8531，大幅低於其他方法，明顯懲罰了不必要的特徵。

### 7.3 特徵得票結果

投票機制綜合了五種方法的選取結果：

| 特徵 | 得票數 | 狀態 |
|------|:---:|------|
| **R&D Spend** | **5/5** | 全票通過 — 無爭議的核心預測因子。所有方法一致認可其預測價值。 |
| **Marketing Spend** | 3/5 | 多數共識 — 具有邊際預測貢獻，但在某些嚴格條件下可能不顯著。 |
| State_Florida | 2/5 | 弱證據 — 僅 RFE 和 Lasso 選取，統計檢定方法不認可其顯著性。 |
| Administration | 1/5 | 噪音 — 僅 Lasso 保留（可能因 L1 懲罰不足），其他方法一致否決。 |
| State_New York | 1/5 | 噪音 — 同 Administration，不具備可靠的預測價值。 |

### 7.4 循序特徵新增分析 (Sequential Feature Addition)

這項分析按照特徵與 Profit 的相關性從高到低的順序逐步加入特徵，追蹤模型性能的變化軌跡。這能幫助我們識別「邊際效益遞減」的臨界點——超過該點後，新增特徵不僅無法改善模型，反而會因過擬合而降低泛化性能。

| 特徵數 | 新增特徵 | RMSE | R² | 變化 |
|:---:|------|---:|---:|------|
| 1 | R&D Spend | $8,274.87 | 0.9465 | 基準模型 — 單一 R&D 支出已能解釋 94.65% 的利潤變異 |
| **2** | **+ Marketing Spend** | **$8,198.80** | **0.9474** | **最佳模型** — RMSE 下降 $76，R² 微幅提升至 0.9474 |
| 3 | + State_Florida | $8,376.45 | 0.9451 | 性能開始下降 — RMSE 上升 $178，R² 下降 0.0023 |
| 4 | + Administration | $9,068.54 | 0.9357 | 顯著過擬合 — RMSE 跳升 $692，R² 持續下降 |
| 5 | + State_New York | $9,137.99 | 0.9347 | 最差性能 — 全模型反而有最高的誤差和最低的解釋力 |

**Elbow Point 分析**：特徵數 = 2 處是明顯的「肘點」（Elbow Point）——在此之前，每增加一個特徵都帶來性能提升或至少不下降；在此之後，每增加一個特徵都導致性能惡化。這是一個極具說服力的證據，證明 R&D + Marketing 是最優特徵組合。

**為什麼更多特徵反而更差？** 這是一個典型的偏差-變異權衡（Bias-Variance Tradeoff）問題：
- 更多的特徵讓模型具有更高的「容量」去擬合訓練資料（降低偏差）
- 但同時增加了模型對訓練資料中隨機雜訊的敏感度（增加變異）
- 當新增特徵的「真實訊號」不足以抵消其引入的「額外變異」時，測試集性能就會下降
- State_Florida、Administration、State_New York 與 Profit 的實質關聯性不足，它們的加入主要為模型帶來了雜訊而非訊號

### 7.5 精煉模型比較

通過去除高影響點和應用穩健估計方法，我們獲得了顯著的性能提升：

| 模型 | 描述 | 測試 R² | Adj. R² | MAE | RMSE |
|------|------|:---:|:---:|---:|---:|
| **Model A** | 移除 3 離群值 + OLS | **0.9601** | **0.9487** | $4,061 | $5,450 |
| Model B | Box-Cox 轉換 Profit | 0.9582 | 0.9530 | $3,677 | $5,575 |
| Model C | Huber 穩健迴歸 | 0.9568 | 0.9445 | $4,280 | $5,668 |

**關鍵發現**：
- 移除僅 3 個高影響點後，R² 從 0.9474 大幅提升至 0.9601，RMSE 從 $8,199 降至 $5,450（降幅達 33.5%）。這說明原始資料中少數幾個異常值對模型性能有不成比例的負面影響。
- 所有三種精煉策略都產生了優於基礎模型的性能，驗證了穩健性處理的必要性。

### 7.6 最終迴歸方程式

以下為各模型的迴歸方程式，可用於手動計算或部署至簡單的決策支援系統：

**Model A（移除離群值後 — 最佳模型）**：
```
Profit = 56,713.92 + 0.7649 × R&D_Spend + 0.0287 × Marketing_Spend
```

解讀：在控制行銷支出的情況下，每增加 $1 的研發支出，預期利潤增加約 $0.76。在控制研發支出的情況下，每增加 $1 的行銷支出，預期利潤增加約 $0.029。

**基礎 OLS（全資料集）**：
```
Profit = 49,032.00 + 0.854 × R&D_Spend
```

解讀：在不考慮其他因素時，每增加 $1 研發支出，預期利潤增加約 $0.85。

**標準化尺度（Z-score）**：
```
Profit = 111,199.31 + 38,698.52 × Z_R&D_Spend
```

解讀：R&D 支出每增加一個標準差（約 $45,902），預期利潤增加約 $38,699。這進一步說明了 R&D 支出的巨大影響力。

---

## 8. 模型診斷

模型訓練完成後，必須對其進行全面的診斷，以驗證線性迴歸的關鍵假設是否成立。這些診斷不僅確認模型的有效性，還可能揭示資料中隱藏的問題（離群值、非常態性、異質變異等），指導後續的精煉策略。

### 8.1 高斯-馬可夫假設檢驗

OLS 估計量的優良性質（不偏性、最小變異）依賴於以下假設的成立程度：

| 假設 | 檢驗方法 | 違反後果 |
|------|---------|---------|
| 線性性（Linearity） | 殘差 vs 擬合值圖 | 模型設定錯誤，預測系統性偏差 |
| 殘差獨立性 | Durbin-Watson 檢定 | 標準誤低估，P-value 過度樂觀 |
| 殘差常態性 | Omnibus / Jarque-Bera 檢定 | P-value 與信賴區間不準確（在小樣本中尤其嚴重） |
| 同質變異性（Homoscedasticity） | Breusch-Pagan 檢定、殘差 vs 擬合值圖 | 標準誤估計偏差，預測區間不準確 |
| 無完全共線性 | VIF（Variance Inflation Factor） | 係數估計不穩定，標準誤極大 |

### 8.2 基礎診斷（全資料集，n=50）

使用完整 50 筆資料的診斷結果：

| 診斷指標 | 數值 | 判定 | 說明 |
|----------|------|------|------|
| Cook's Distance | 3 點 > 4/n (0.08) | 需處理 | 三個觀測值對模型係數有不成比例的高影響，可能扭曲估計 |
| Omnibus p-value | 0.0010 | 殘差非常態 | 拒絕殘差常態性的虛無假設，可能影響 P-value 的準確性 |
| Jarque-Bera p-value | < 0.05 | 殘差非常態 | 基於偏態與峰度的常態性檢定，與 Omnibus 結論一致 |
| Durbin-Watson | 1.28 | 輕微正自相關 | 理想值為 2.0，1.28 暗示殘差可能有輕微的正自相關（相鄰觀測值的殘差趨向同號） |
| VIF (max) | 2.47 (R&D) | 無嚴重共線性 | VIF < 10 通常被認為是可接受的。R&D 雖有最高的 VIF，但仍遠低於警戒線 |

### 8.3 清洗後診斷（移除 3 離群值，n=47）

移除三個高影響點後，診斷指標大幅改善：

| 診斷指標 | 數值 | 判定 | 說明 |
|----------|------|------|------|
| Omnibus p-value | 0.831 | 殘差服從常態 | p > 0.05，無法拒絕常態性虛無假設 — 殘差呈常態分佈 |
| Jarque-Bera p-value | 0.773 | 殘差服從常態 | 偏態與峰度檢定均不顯著，確認常態性成立 |
| Durbin-Watson | 2.074 | 無自相關 | 接近理想值 2.0，殘差之間無系統性相關 |
| Marketing Spend P-value | 0.037 | 統計顯著 | 在 α = 0.05 標準下，行銷支出變得顯著（原來在全資料集中不顯著） |
| Adjusted R² | 0.960 | 優秀 | 模型解釋了 96% 的調整後利潤變異 |

**重要觀察**：移除僅 3 個離群值後，殘差診斷從「問題嚴重」變為「全部過關」。這說明原始資料中的少數異常點是診斷問題的主要來源，而非模型本身的設定錯誤。同時，Marketing Spend 在清洗後變得顯著，暗示離群值可能「遮蔽」了行銷支出的真實訊號。

### 8.4 交叉驗證穩定性

使用 5-fold 交叉驗證評估不同特徵組合的泛化穩定性：

| 特徵組合 | 5-fold CV R² Mean | Std |
|----------|:---:|:---:|
| R&D only | 0.9374 | 0.0373 |
| R&D + Marketing | 0.9389 | 0.0373 |
| R&D + Admin | 0.9304 | 0.0393 |

- R&D + Marketing 的 CV R²（0.9389）略高於 R&D only（0.9374），但差異非常小（僅 0.0015），且標準差相同。這說明 Marketing 的邊際貢獻雖為正向，但幅度有限。
- R&D + Admin 的 CV R²（0.9304）反而低於 R&D only，驗證了 Administration 是噪音變數的結論。

### 8.5 各州 R&D → Profit 斜率一致性

為了驗證「State 變數不顯著」的結論，我們進一步檢查了三個州中 R&D 與 Profit 的線性關係是否一致：

| State | 斜率 | 州內 r | 樣本數 |
|------|:---:|:---:|:---:|
| California | 0.928 | 0.975 | 17 |
| New York | 0.813 | 0.976 | 17 |
| Florida | 0.813 | 0.970 | 16 |

三州的 R&D-Profit 斜率高度一致（0.813–0.928），且州內相關係數都極高（≥ 0.970）。這提供了強有力的證據：R&D 對利潤的影響在不同州之間沒有實質差異，因此沒必要為每個州建立單獨的模型。State 變數之所以在統計上不顯著，正是因為其效應已被 R&D 完全捕捉。

### 8.6 診斷失敗的應對策略

當殘差診斷顯示假設違反時，不應立即放棄模型。以下是一個系統性的問題處理框架：

**問題 1：殘差非常態（Omnibus p < 0.05）**

殘差非常態是迴歸分析中最常見的問題之一。可能的解決方案按照優先順序排列：

1. **檢查離群值**：少數極端值可能是非常態性的來源。使用 Cook's Distance 識別並評估是否應移除。在本專案中，移除 3 個高影響點就使得 Omnibus p 從 0.0010 改善至 0.831。
2. **轉換目標變數**：對 Profit 進行 Box-Cox 或對數轉換（log transformation）可以有效地使右偏分佈變得更接近常態。轉換後需要對預測值進行反轉換以回到原始尺度。
3. **使用穩健標準誤**：如果殘差非常態但樣本量足夠大（n > 100），可以依賴中央極限定理，使用 Heteroscedasticity-Consistent (HC) 標準誤來獲得有效的 P-value 和信賴區間。在 statsmodels 中使用 `model.fit(cov_type='HC3')` 來啟用。
4. **切換到非參數方法**：如果上述方法都無效，考慮使用不需要常態性假設的方法，如 Quantile Regression（分位數迴歸）或 Bootstrap 重抽樣法。

**問題 2：殘差自相關（Durbin-Watson 遠離 2.0）**

殘差自相關表示觀測值之間不是獨立的——一個觀測值的殘差與另一個觀測值的殘差有系統性的關聯。對於橫斷面資料（如本專案），自相關通常不是主要問題。但如果在時序資料中出現，應考慮：
- 加入時間趨勢變數（如 year、month）作為控制變數
- 使用 ARIMA 或 GLS（Generalized Least Squares）等專門處理自相關的模型
- 使用 Cochrane-Orcutt 程序進行迭代修正

**問題 3：異質變異性（殘差散佈隨預測值增大而增大）**

異質變異性（Heteroscedasticity）表現為殘差圖中的「漏斗形」——預測值較大處的殘差離散程度明顯大於預測值較小處。解決方案：
- 對目標變數進行對數轉換（常能同時解決非常態性和異質變異性）
- 使用加權最小平方法（Weighted Least Squares, WLS），給予變異較小的觀測值更大的權重
- 如同非常態性，使用穩健標準誤（HC3）

**問題 4：多重共線性（VIF > 10）**

如果發現 VIF 超過 10 的特徵，表示存在嚴重的多重共線性：
- 移除其中一個高度相關的特徵（如 R&D 和 Marketing 同時存在時）
- 使用 Ridge 迴歸（L2 正則化）替代 OLS——Ridge 能處理共線性，但不會自動淘汰特徵
- 使用主成分迴歸（PCR）或偏最小平方法（PLS）：先將高度相關的特徵投影到較低維度的主成分空間，再用主成分作為新的預測變數

### 8.7 模型診斷的全面檢核表

以下是一個可以在任何迴歸專案中重複使用的診斷檢核表。建議在每次建模後逐項檢查：

| 編號 | 檢核項目 | 理想標準 | 不通過時的行動 |
|:---:|---------|---------|--------------|
| D1 | 模型整體 F 檢定 | P-value < 0.05 | 模型可能需要不同的特徵組合或非線性變換 |
| D2 | 各特徵係數 t 檢定 | P-value < 0.05（或根據業務需求放寬）| 考慮移除不顯著特徵或收集更多資料 |
| D3 | R² 與 Adjusted R² 差距 | Adj. R² 接近 R²（差距 < 0.1） | 差距大表示有過多不必要特徵 |
| D4 | 殘差常態性（Omnibus） | P-value > 0.05 | 檢查離群值 > Box-Cox 轉換 > 穩健標準誤 |
| D5 | 殘差自相關（Durbin-Watson） | 1.5 < DW < 2.5 | 檢查資料排序 > 加入時間變數 > GLS |
| D6 | 殘差異質變異性（圖形檢測） | 殘差 vs 擬合值圖中無明顯漏斗形 | 對數轉換 > WLS > 穩健標準誤 |
| D7 | 多重共線性（VIF） | 所有 VIF < 10（最好 < 5） | 移除高 VIF 特徵 > Ridge 迴歸 > 降維 |
| D8 | 高影響點（Cook's D） | 無點超過 4/n | 逐一檢查高影響點 > 考慮移除或使用穩健方法 |
| D9 | 交叉驗證穩定性 | CV R² Std < 0.05 | 收集更多資料 > 簡化模型 > 正則化 |
| D10 | 預測誤差合理性 | RMSE < 業務可接受範圍 | 與領域專家確認 > 考慮非線性模型 |



---

## 9. 部署架構

### 9.1 Streamlit Cloud 互動式儀表板

Streamlit 儀表板是本專案的旗艦部署形式，提供完整的互動式資料探索與分析體驗。

- **URL**：[l6-new-model.streamlit.app](https://l6-new-model.streamlit.app/)
- **原始碼**：`app.py`
- **部署方式**：Streamlit Cloud 透過 GitHub OAuth 連接本倉庫，在每次推送至 `master` 分支時自動重新部署。
- **資料內嵌**：所有 50 筆原始資料和循序特徵新增分析結果均以 Python 列表形式 Hardcode 於 `app.py` 中，不需依賴外部 CSV 檔案。這確保了應用程式在任何環境中都能獨立執行，無需檔案系統存取權限。

**儀表板結構（4 章節）**：

| 章節 | 內容 | 關鍵互動元件 |
|------|------|-------------|
| **Project Overview** | 專案使命（Mission Statement）、資料集規模摘要（50 筆 × 5 欄）、核心目標 | 資訊區塊、Metric 卡片（資料筆數、特徵數、目標變數） |
| **Data Discovery** | 原始數據統計摘要（describe）、KPI 展示（平均 R&D、平均 Marketing、平均 Profit） | Checkbox 展開/隱藏 50 筆完整表格、3 欄 Metric 卡 |
| **CRISP-DM Workflow** | 4 步驟管線（Data Cleaning → Feature Encoding → Train-Test Split → MLR Modeling） | st.tabs 頁籤切換，每個頁籤含文字說明 + 程式碼區塊 |
| **Feature Selection Analysis** | SFA 結果互動表格、雙線聯動圖（RMSE 與 R²）、最佳模型洞察 | Plotly 互動圖（hover 顯示特徵組合、縮放、平移）、st.dataframe 格式化表格 |

**圖表互動性**：使用 Plotly Express 繪製的折線圖支援以下互動操作：
- **Hover**：將滑鼠移到任一資料點上，顯示該模型的特徵組合、RMSE 值與 R² 值。
- **Zoom**：拖曳選取圖表區域進行放大，便於檢視特定區間的細節。
- **Pan**：在放大後拖曳平移視角。
- **最佳點標記**：RMSE 最低點（特徵數 = 2）以紅色星號和「BEST」文字標記，讓使用者一眼就能識別最優模型位置。

### 9.2 GitHub Pages 靜態儀表板

除了互動式 Streamlit 應用程式外，本專案還部署了一個靜態的 GitHub Pages 儀表板，適合快速瀏覽分析結果。

- **URL**：`https://miccowang66-max.github.io/L6-new-model/`
- **觸發**：推送至 `master` 分支時，GitHub Actions 自動觸發部署。
- **技術**：純靜態 HTML + 手繪 Excalidraw 風格（純 CSS，無外部 UI 框架，使用 Google Fonts 手寫字體）。
- **內容**：8 個資訊圖表區塊，包含 Hero 區塊（關鍵指標卡片）、專案概述、CRISP-DM 工作流程圖、核心建模結果、5 種特徵選取共識表、互動式循序特徵新增圖（可切換 R² / RMSE）、Streamlit 儀表板線框圖與技術棧、關鍵發現。
- **雙頁籤切換**：頁面支援「💻 Interactive Infographic」與「🖼️ Static Poster (PNG)」雙頁籤。後者顯示 `ml_pipeline_infographic.png` 高解析度靜態海報並提供下載按鈕。

| 區塊 | 內容 |
|------|------|
| **Hero / Header** | 4 張指標卡（R² 0.9474、RMSE $8,199、2 最佳特徵、5/5 演算法） + 作者/版本中繼資料 |
| **1. Project Overview** | 專案背景、CRISP-DM 方法論簡介、工作區交付物列表 |
| **5. CRISP-DM Workflow** | 6 步驟流程圖（Business Und. → Data Und. → Data Prep. → Modeling → Evaluation → Deployment） |
| **2. Core Modeling Results** | HTML 表格：Model 1–4 的特徵組合、Test R²、Test RMSE，含最佳模型高亮 |
| **3. Feature Selection Consensus** | HTML 投票矩陣表：5 方法對 5 特徵的排名，含共識洞察 |
| **4. Sequential Feature Addition** | 互動式 SVG 圖（hover 顯示細節），可切換 Test R² / Test RMSE 曲線，含動畫繪製效果 |
| **6 & 7. Dashboard & Tech Stack** | Streamlit 儀表板線框圖 + Python / scikit-learn / Plotly 等技術棧徽章 |
| **8. Key Findings** | 三個核心戰略洞察（R&D 為王、行銷邊際價值、拒絕行政雜訊）+ 火箭插圖 |

### 9.3 CI/CD 管線

GitHub Actions 工作流程定義在 `.github/workflows/deploy.yml`，自動化整個部署過程：

```yaml
觸發條件：push to master
├── checkout 原始碼（actions/checkout@v3）
├── configure-pages（設定 GitHub Pages 環境）
├── upload-pages-artifact（上傳靜態檔案至 GitHub Artifacts）
└── deploy-pages（部署至 GitHub Pages CDN 全球邊緣節點）
```

整個部署過程通常在 30–60 秒內完成，無需手動操作。

### 9.4 Agent Skill 發佈

本專案的完整方法論已封裝為可複用的 OpenCode Agent Skill，使得 AI 輔助開發工具能在其他專案中自動執行相同的分析流程。

- **路徑**：`.opencode/skills/ml-regression-pipeline/SKILL.md`
- **觸發條件**：當使用者提及以下任一關鍵詞時，Agent 會自動載入此 Skill：
  - `regression analysis`
  - `feature selection`
  - `backward elimination` / `forward selection`
  - `RFE` / `Lasso`
  - `mutual information`
- **Skill 涵蓋**：完整的 16 步驟工作流程，包括架構設計指南、5 種方法的 Python 程式碼樣板、診斷檢查表、GitHub Pages 儀表板部署腳本、白皮書生成模板、README 格式規範。
- **可複用性**：在新專案中，Agent 可以根據此 Skill 自動產生客製化的分析腳本，根據新資料集的特性調整參數（如顯著水準 α、測試集比例、交叉驗證折數），同時保持方法論的一致性。

---

## 10. 快速入門與實作教學

本章提供一個從零開始的完整實作教學，帶領讀者在自己的環境中複製本專案的所有分析結果。無論你是資料科學初學者還是有經驗的從業者，遵循以下步驟都可以在 30 分鐘內完成從安裝到完整分析的流程。

### 10.1 環境設置

**前置需求**：
- Python 3.10 或更高版本（建議使用 Python 3.11 以獲得最佳效能）
- Git（用於複製倉庫）
- 至少 1 GB 可用硬碟空間
- macOS、Windows 10+ 或 Linux 作業系統

**步驟 1：複製倉庫**

打開終端機（Terminal），執行以下命令將整個專案下載到本地：

```bash
git clone https://github.com/miccowang66-max/L6-new-model.git
cd L6-new-model
```

這個命令會在工作目錄中創建一個名為 `L6-new-model` 的資料夾，其中包含所有源碼、資料和設定檔。

**步驟 2：建立虛擬環境（強烈建議）**

虛擬環境可以將本專案的相依套件與系統全域的 Python 套件隔離，避免版本衝突：

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

當終端機提示字元前面出現 `(venv)` 字樣時，表示虛擬環境已成功啟動。

**步驟 3：安裝相依套件**

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含了所有必要的 Python 套件及其最低版本要求。安裝過程通常需要 2–5 分鐘，取決於網路速度和系統效能。

如果你只需要執行互動式 Streamlit 儀表板（而非完整分析管線），至少需要安裝以下套件：

```bash
pip install streamlit pandas numpy plotly
```

**步驟 4：驗證安裝**

執行以下命令確認所有套件已正確安裝且版本相容：

```bash
python -c "import pandas, numpy, sklearn, statsmodels, matplotlib, seaborn, streamlit, plotly; print('All imports successful!')"
```

如果看到 "All imports successful!" 且沒有任何錯誤訊息，表示環境已就緒。

### 10.2 執行完整分析管線

本專案包含多個獨立的 Python 腳本，每個腳本負責不同的分析階段。以下按照建議的執行順序說明：

**腳本 1：基礎模型管線（main_analysis.py）**

這是核心腳本，執行完整的 CRISP-DM 流程。它會自動完成以下任務並在終端機輸出詳細的過程日誌：

```bash
python main_analysis.py
```

執行過程中你會看到：
1. **Part 1**：資料載入、相關性分析、熱力圖與散佈圖的生成
2. **Part 2**：One-Hot Encoding、虛擬變數陷阱避免、標準化、Train/Test Split
3. **Part 3**：Backward Elimination 的逐步淘汰過程（每輪顯示各特徵的 P-value 和被移除的特徵）
4. **Part 4**：最終模型評估（R²、Adjusted R²、MAE、RMSE）、預測 vs 實際圖、殘差圖

所有輸出的圖表會儲存在 `outputs/figures/`，報告儲存在 `outputs/reports/`，前處理後的資料儲存在 `data/processed/`。

**腳本 2：五種特徵選取方法比較（feature_selection.py）**

```bash
python feature_selection.py
```

這個腳本會：
- 執行全部五種特徵選取方法（Backward Elimination、Forward Selection、RFE、Lasso、Mutual Information）
- 計算投票結果（每個特徵獲得的票數）
- 生成特徵選取比較圖（熱力圖 vs R² 長條圖）
- 輸出詳細的文字報告至 `outputs/reports/feature_selection.txt`

**腳本 3：精煉模型分析（refined_models.py）**

```bash
python refined_models.py
```

這個腳本會自動測試多種模型精煉策略：
- Model A：移除高影響點（Cook's D > 4/n）後重新訓練
- Model B：應用 Box-Cox 轉換使殘差服從常態分佈
- Model C：使用 Huber 穩健迴歸降低離群值影響
- 生成模型比較圖與詳細報告

**腳本 4：循序特徵新增圖表（outcome_visualization.py）**

```bash
python outcome_visualization.py
```

生成特徵數（1→5）vs RMSE 和 R² 的雙線圖，以及用於 Excel 的 CSV/TSV 數據表。

**腳本 5：方法比較圖表（method_comparison_charts.py）**

```bash
python method_comparison_charts.py
```

生成五種特徵選取方法的綜合比較圖表，包括水平長條圖（RMSE 和 R²）、熱力圖（特徵選取矩陣）、以及 4 合 1 全覽儀表板。

**腳本 6：補充診斷分析（supplement_analysis.py）**

```bash
python supplement_analysis.py
```

執行進階診斷，包括 VIF 共線性分析、5-fold 交叉驗證、Cook's Distance 影響點分析、以及各州 R&D-Profit 斜率一致性檢定。

### 10.3 啟動 Streamlit 互動式儀表板

這是整個專案中最具互動性的部分，提供一個可以在瀏覽器中操作的圖形化介面：

```bash
streamlit run app.py
```

執行後，Streamlit 會自動在預設瀏覽器中打開一個新頁面（通常是 http://localhost:8501）。儀表板包含以下四個章節：

1. **Project Overview**：專案使命與關鍵指標一覽
2. **Data Discovery**：可展開/隱藏的原始資料表、統計摘要、KPI 指標卡
3. **CRISP-DM Modeling Workflow**：四個頁籤展示從資料清洗到模型建模的完整管線
4. **Advanced Feature Selection Analysis**：互動式 SFA 數據表與雙線聯動圖（hover 可查看特徵組合詳情）

**Streamlit 儀表板的互動操作指南**：
- **展開/隱藏表格**：勾選 "Show / Hide Raw Data" 的 checkbox
- **切換頁籤**：點擊 CRISP-DM Workflow 區域的不同頁籤查看各步驟
- **Hover 圖表**：將滑鼠移到折線圖的資料點上查看具體數值
- **縮放圖表**：在圖表上拖曳選取區域進行放大
- **雙擊重置**：雙擊圖表可以恢復到預設視角

### 10.4 自訂分析：修改參數指南

如果你想根據自己的需求調整分析參數，以下是指南：

**修改顯著水準 α**：

在 `main_analysis.py` 中找到以下行並修改數值：

```python
SIGNIFICANCE_LEVEL = 0.05  # 改為 0.01 會更嚴格，0.10 會更寬鬆
```

較小的 α（如 0.01）會使模型更保守，傾向於保留較少的特徵；較大的 α（如 0.10）會使模型更寬容，傾向於保留較多的特徵。

**修改 Train/Test 分割比例**：

```python
TEST_SIZE = 0.2   # 改為 0.3 表示 70/30 分割
RANDOM_STATE = 0  # 改為其他整數以獲得不同的分割結果
```

**更換資料集**：

1. 將你的 CSV 檔案放入 `data/raw/` 目錄
2. 在 `main_analysis.py` 中修改 `RAW_DATA` 變數指向新檔案
3. 修改 `TARGET` 變數為你的目標欄位名稱

```python
RAW_DATA = "data/raw/your_dataset.csv"
TARGET = "YourTargetColumn"
```

### 10.5 常見執行問題與解決方案

| 問題 | 可能原因 | 解決方案 |
|------|---------|---------|
| `ModuleNotFoundError: No module named 'statsmodels'` | 相依套件未安裝 | 執行 `pip install -r requirements.txt` |
| `FileNotFoundError: data/raw/50_startups.csv` | 不在正確的工作目錄 | 確保在 `L6-new-model/` 目錄下執行腳本 |
| `Permission denied` | 輸出目錄權限不足 | 手動創建 `outputs/` 和 `data/processed/` 目錄 |
| Streamlit 無法啟動 | 埠號 8501 被佔用 | 使用 `streamlit run app.py --server.port 8502` 更換埠號 |
| 中文在圖表中顯示為方框 | 系統缺少中文字型 | Windows 通常已內建；macOS/Linux 需安裝 Noto Sans CJK 或類似字型 |
| `MemoryError` | 資料集過大（非本專案問題） | 對於 >10,000 筆的資料集，考慮使用 `chunksize` 分批讀取 |

### 10.6 深入了解輸出結果

執行完分析管線後，以下是對主要輸出的詳細解讀指南，幫助你最大化利用分析結果。

**outputs/figures/ 目錄中的圖表解讀**：

1. **corr_heatmap.png**（相關性熱力圖）：
   - 顏色越紅表示正相關越強，越藍表示負相關越強
   - 重點關注目標變數 Profit 所在的行/列
   - R&D Spend 與 Profit 的相關係數（0.97）應該是最顯眼的數值
   - 特徵之間的相關性（如 R&D 與 Marketing 之間）如果過高（>0.9），可能表示存在多重共線性問題

2. **scatter_features.png**（散佈圖）：
   - R&D vs Profit 應呈現清晰的線性趨勢（點沿一條直線緊密排列）
   - Admin vs Profit 應呈現較為鬆散的分佈（沒有明顯趨勢）
   - 散佈圖中的離群點（遠離主體的點）對應於 Cook's D 分析中的高影響點

3. **feature_count_performance.png**（特徵數 vs 性能）：
   - RMSE 線在特徵數 = 2 時應該處於最低點
   - R² 線在特徵數 = 2 時應該處於最高點或次高點
   - 特徵數 > 2 後兩條線的趨勢應該相反（RMSE 上升、R² 下降），這是過擬合的經典信號

4. **pred_vs_actual.png**（預測 vs 實際）：
   - 理想情況下，點應該緊密圍繞紅色虛線（y = x）
   - 點越靠近紅線，表示預測越準確
   - 如果點呈現系統性的偏離（如全部在紅線上方或下方），表示模型可能有偏差

5. **residuals.png**（殘差圖）：
   - 左圖（殘差 vs 預測值）：點應該隨機分佈在 y = 0 線的兩側，不應該有漏斗形或 U 形
   - 右圖（殘差直方圖）：應該呈現鐘形（bell-shaped）常態分佈

**outputs/reports/ 目錄中的報告解讀**：

- **metrics.txt**：這是 statsmodels OLS 的完整輸出。重點查看以下部分：
  - `R-squared` 和 `Adj. R-squared`：模型整體解釋力
  - `coef` 欄：每個特徵的係數（邊際效應）
  - `P>|t|` 欄：每個係數的 P-value（< 0.05 表示顯著）
  - `Omnibus` 和 `Durbin-Watson`：殘差診斷指標
  - `Cond. No.`：條件數，如果非常大（> 30）表示可能存在共線性

- **feature_selection.txt**：比較各方法的選取結果。關注：
  - 哪些特徵被所有方法選取（穩健的核心特徵）
  - 哪些特徵僅被部分方法選取（情境依賴的邊際特徵）
  - 哪些特徵被所有方法淘汰（明確的噪音變數）

- **method_comparison_results.csv**：用 Excel 開啟此檔案，可以使用排序和篩選功能進行交互式探索。建議按照 RMSE 升冪排序，識別最佳方法；按照特徵數（n_Feat）排序，分析複雜度與性能的權衡。

### 10.7 在 Jupyter Notebook 中進行互動式探索

如果你偏好逐步執行分析而非一次執行整個腳本，可以將程式碼遷移到 Jupyter Notebook 中。以下是一個建議的 Notebook 結構：

**Cell 1 — 環境設置**：
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE, mutual_info_regression

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
sns.set_style("whitegrid")
```

**Cell 2 — 資料載入與初步探索**：
```python
df = pd.read_csv("data/raw/50_startups.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values:\n{df.isnull().sum()}")
df.describe()
```

**Cell 3 — 相關性分析**：
```python
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0)
plt.title("Pearson Correlation Matrix")
plt.show()
```

**Cell 4 — 前處理**：
```python
# One-Hot Encoding
dummies = pd.get_dummies(df["State"], prefix="State", dtype=int)
dummies = dummies.drop(columns=[dummies.columns[0]])  # 避免 Dummy Trap
df_encoded = pd.concat([df.drop(columns=["State"]), dummies], axis=1)

# Feature/Target split
X = df_encoded.drop(columns=["Profit"])
y = df_encoded["Profit"]

# Standardization
scaler = StandardScaler()
num_cols = ["R&D Spend", "Administration", "Marketing Spend"]
X[num_cols] = scaler.fit_transform(X[num_cols])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
```

**Cell 5 — 模型訓練與評估**：
```python
# Backward Elimination
features = list(X_train.columns)
X_be = sm.add_constant(X_train[features])
model = sm.OLS(y_train, X_be).fit()
print(model.summary())
```

這種 Notebook 方式的優點是可以逐 Cell 執行、即時查看中間結果、以及在每個步驟後進行額外的探索性分析。對於學習和理解整個管線非常有幫助。

### 10.8 使用 Docker 容器化部署（進階）

對於需要在不同環境中部署或希望完全隔離執行環境的使用者，可以使用 Docker 進行容器化。以下是一個完整的 Docker 設定指南。

**Dockerfile 範例**：

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安裝系統相依套件（matplotlib 需要）
RUN apt-get update && apt-get install -y \\
    libfreetype6-dev \\
    libpng-dev \\
    && rm -rf /var/lib/apt/lists/*

# 複製並安裝 Python 相依套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案檔案
COPY . .

# Streamlit 預設埠號
EXPOSE 8501

# 預設執行 Streamlit 儀表板
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**建置與執行**：

```bash
# 建置 Docker 映像檔
docker build -t l6-profit-predictor .

# 執行容器（Streamlit 儀表板模式）
docker run -p 8501:8501 l6-profit-predictor

# 執行容器（完整分析管線模式）
docker run -v $(pwd)/outputs:/app/outputs l6-profit-predictor python main_analysis.py
```

第一個命令會啟動 Streamlit 儀表板，可在瀏覽器中訪問 `http://localhost:8501`。第二個命令會在容器內執行完整的分析管線，並將輸出的圖表和報告保存到宿主機的 `outputs/` 目錄中（透過 volume 掛載）。

**Docker Compose（多服務配置）**：

如果需要在同一環境中同時執行 Streamlit 儀表板和定期的分析管線更新，可以使用 Docker Compose：

```yaml
version: '3.8'
services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    command: streamlit run app.py --server.port=8501 --server.address=0.0.0.0
    restart: unless-stopped

  analysis:
    build: .
    volumes:
      - ./outputs:/app/outputs
      - ./data/processed:/app/data/processed
    command: python main_analysis.py
    profiles:
      - analysis  # 僅在指定 profile 時執行
```

使用方式：
```bash
# 啟動儀表板（持續執行）
docker compose up dashboard -d

# 執行一次性分析
docker compose --profile analysis run analysis
```

### 10.9 專案維護與更新指南

隨著 Python 生態系統的演進，相依套件可能會有重大更新。以下是指南幫助你保持專案的可執行性：

**定期檢查套件更新**：
```bash
pip list --outdated
```

這個命令列出所有有新版本可用的已安裝套件。重點關注 `scikit-learn`、`statsmodels` 和 `pandas` 的更新，因為這些套件的 API 變化可能影響分析結果的重現性。

**鎖定相依版本**（用於生產環境）：
```bash
pip freeze > requirements-lock.txt
```

`requirements-lock.txt` 會記錄每個套件的精確版本（含次版本號），確保在任何環境中安裝的套件版本完全一致。在開發環境中使用 `requirements.txt`（含最低版本約束），在生產環境中使用 `requirements-lock.txt`（含精確版本）。

**更新資料集時的注意事項**：
1. 將新的 CSV 檔案放入 `data/raw/`，保留舊檔案作為備份
2. 更新 `main_analysis.py` 中的 `RAW_DATA` 路徑變數
3. 檢查新資料的欄位名稱是否與程式碼中的參考一致（特別是 `State` 和 `Profit`）
4. 執行 `python main_analysis.py` 以重新生成所有圖表和報告
5. 比較新舊結果的 R² 和 RMSE，評估資料更新對模型性能的影響

**將專案用於不同領域資料的指南**：

這個分析框架不僅限於新創公司利潤預測，可以應用於任何類似結構的資料集（多個連續特徵 + 可選類別特徵 → 預測一個連續目標）。以下是一些潛在的應用場景：

| 領域 | 目標變數 | 潛在預測特徵 |
|------|---------|------------|
| 房地產 | 房價 | 面積、房間數、屋齡、地段（類別） |
| 醫療保健 | 住院天數 | 年齡、併發症數量、入院檢驗值、科別（類別） |
| 教育 | 學生成績 | 出席率、作業完成率、課外活動時數、班級（類別） |
| 行銷 | 客戶終身價值 | 購買頻率、平均訂單金額、退貨率、會員等級（類別） |
| 人力資源 | 員工薪資 | 年資、績效評分、教育程度、部門（類別） |

移植步驟：
1. 準備好你的 CSV 檔案，確保有明確的目標欄位（連續數值）
2. 修改 `RAW_DATA` 為新檔案路徑、`TARGET` 為新目標欄位名稱
3. 檢查類別變數並調整 One-Hot Encoding 的欄位名稱
4. 考慮是否需要調整 `SIGNIFICANCE_LEVEL`（小樣本時放寬至 0.10，大樣本時收緊至 0.01）
5. 根據領域知識調整精煉模型的策略（例如，若領域中離群值很重要，可能不應移除）

---

## 11. 商業解讀與決策指南

本章將統計分析的結果轉化為可供商業決策者直接使用的洞察與建議。技術指標（如 R²、P-value、RMSE）在此被翻譯為商業語言，幫助非技術背景的利害關係人理解模型告訴我們什麼。

### 11.1 核心商業洞察

**洞察 1：R&D 支出是利潤的絕對主導因素**

統計證據非常明確：R&D Spend 是預測新創公司利潤的最重要因素（Pearson r = 0.973，全票通過 5 種特徵選取方法）。這意味著，在所有可控的營運支出中，**研發投入對獲利能力的影響遠超過其他任何支出類別**。

商業含義：
- 當面臨預算限制時，應優先保障 R&D 支出的充足性
- 削減 R&D 預期會對利潤產生不成比例的負面影響
- 投資者在評估新創公司時，應將 R&D 支出水平作為首要的盡職調查指標

**洞察 2：行銷支出具有邊際但正向的貢獻**

Marketing Spend 在 5 種方法中獲得 3/5 的支持率（多數共識）。加入行銷支出後，RMSE 從 $8,275 降至 $8,199（減少約 0.92%），R² 從 0.9465 微升至 0.9474。效果雖然不大，但確實是正向的。

商業含義：
- 行銷支出對利潤有正向貢獻，但邊際效益遠小於 R&D
- 每增加 $1 行銷支出，預期利潤增加約 $0.029（基於 Model A 方程）
- 相較之下，每增加 $1 研發支出，預期利潤增加約 $0.765
- **R&D 的投資回報率（ROI）是行銷的約 26 倍**

**洞察 3：行政支出和地理位置不具預測價值**

Administration、State_Florida 和 State_New York 在五種特徵選取方法中分別僅獲得 1/5、2/5 和 1/5 的票數。這些變數對利潤的預測不具統計顯著性，加入模型反而會降低預測準確度。

商業含義：
- 行政支出的增減不太可能對利潤產生可預測的影響（至少在本資料集的範圍內）
- 公司的地理位置（California vs. Florida vs. New York）不是決定利潤的關鍵因素
- 在制定預算分配策略時，不需要針對不同州制定差異化的模型

### 11.2 投資決策框架

基於本分析的結果，我們可以提出一個簡單的投資決策框架：

**框架 1：新創公司利潤預測公式**

使用 Model A（移除離群值後的最優模型）進行利潤預測：

```
預期利潤 = $56,714 + $0.765 × R&D_Spend + $0.029 × Marketing_Spend
```

這個公式可以用於快速估算一家新創公司在特定支出結構下的預期利潤。例如：

- 一家投入 $100,000 研發和 $200,000 行銷的公司，預期利潤約為：
  56,714 + 0.765 × 100,000 + 0.029 × 200,000 = 56,714 + 76,500 + 5,800 = **$139,014**

**框架 2：預算優化策略**

由於 R&D 的邊際回報遠高於 Marketing，最優的預算分配策略是：
1. 確保 R&D 支出達到合理水平（參考行業基準）
2. 如果有額外預算，優先分配到 R&D（直到邊際回報遞減）
3. 行銷支出維持在維持市場存在的必要水平即可
4. 行政支出控制在行業標準範圍內，不需要刻意增加

**框架 3：風險評估**

模型的 RMSE 約為 $5,450（Model A），這提供了一個預測誤差的參考範圍。在進行投資決策時：
- 預測利潤 ± RMSE（即 ± $5,450）可以作為 68% 信賴區間（假設誤差為常態分佈）
- 預測利潤 ± 2 × RMSE（即 ± $10,900）可以作為 95% 信賴區間

### 11.3 模型的限制與適用範圍

任何預測模型都有其局限性。在使用本模型時，請注意以下限制：

**限制 1：樣本代表性**
- 資料集僅包含 50 間美國新創公司，樣本規模有限
- 樣本中的公司可能具有特定的產業或階段特徵（如科技導向、早期階段）
- 推廣到其他產業、其他國家、或更大規模的公司時需謹慎

**限制 2：因果關係的混淆**
- 模型建立的是「相關性」而非「因果性」
- R&D 支出與利潤的高度相關可能部分反映了其他未觀測因素（如公司品質、管理能力）的影響
- 不應簡單地推論「增加 R&D 支出就一定會增加利潤」

**限制 3：時間維度的缺失**
- 資料集是橫斷面數據（cross-sectional），不包含時間維度
- 模型無法捕捉支出與利潤之間的時間延遲效應（例如，今年的 R&D 支出可能影響明年的利潤）
- 不適用於時間序列預測

**限制 4：線性關係假設**
- 模型假設支出與利潤之間的關係是線性的
- 實際上可能存在非線性關係（如 R&D 的邊際回報遞減）
- 對於支出極端值（非常高或非常低）的預測準確度可能較差

### 11.4 面向不同利害關係人的溝通要點

**向投資者匯報時應強調**：
- R&D Spend 是最可靠的利潤預測指標（5/5 方法一致認可）
- Model A 達到 R² = 0.9601，意味著模型解釋了 96% 的利潤變異
- RMSE = $5,450，預測誤差在可接受範圍內

**向營運團隊匯報時應強調**：
- 行政支出對利潤沒有顯著影響，優化行政成本不會顯著改善盈利
- 行銷支出有正向但微小的貢獻，不應過度投資
- 重點應放在研發效率上——如何讓每一塊錢的 R&D 產生最大的利潤回報

**向資料科學團隊匯報時應強調**：
- 5 種特徵選取方法提供了交叉驗證的穩健性
- 移除 3 個高影響點後診斷指標全面通過
- 精煉模型（A/B/C）的結果驗證了基礎分析的結論

### 11.5 案例研究：三間虛構新創公司的模型應用

為了更具體地展示本模型的實際應用價值，以下透過三個虛構的新創公司案例來說明如何利用模型進行利潤預測和預算規劃。三間公司分別代表不同階段的支出結構。

**案例 A：TechGrowth（重研發、輕行銷型）**

TechGrowth 是一間專注於 B2B SaaS 產品的早期新創公司。其年度支出結構如下：
- R&D Spend：$150,000（高研發投入，佔總支出的 86%）
- Marketing Spend：$25,000（低行銷支出，聚焦於內容行銷和 SEO）
- Administration：$80,000（基本行政成本）

模型預測利潤：
```
Profit = 56,714 + 0.765 × 150,000 + 0.029 × 25,000
       = 56,714 + 114,750 + 725
       = $172,189
```

以 95% 信賴區間計算：$172,189 ± 2 × $5,450 = [$161,289, $183,089]

TechGrowth 的預測利潤相當可觀（$172,189），主要歸功於其高額的研發投入。即使行銷支出相對較低，研發支出對利潤的巨大邊際貢獻（每 $1 研發帶來 $0.765 利潤）仍使公司獲得良好的預測結果。這個結果支持 TechGrowth 繼續將大部分預算配置於研發的策略。

**案例 B：MarketFirst（輕研發、重行銷型）**

MarketFirst 是一間 D2C 消費品牌新創公司，資源主要配置於市場推廣和品牌建立。其年度支出結構如下：
- R&D Spend：$30,000（輕研發，主要用於包裝設計和產品改良）
- Marketing Spend：$350,000（重行銷，社群媒體和線上廣告為主力）
- Administration：$120,000

模型預測利潤：
```
Profit = 56,714 + 0.765 × 30,000 + 0.029 × 350,000
       = 56,714 + 22,950 + 10,150
       = $89,814
```

MarketFirst 的預測利潤（$89,814）顯著低於 TechGrowth（$172,189），儘管 MarketFirst 的總營運支出（$500,000）遠高於 TechGrowth（$255,000）。這個鮮明的對比說明了本模型的核心結論：R&D 的利潤轉換效率遠高於 Marketing。MarketFirst 投入了 14 倍於 TechGrowth 的行銷預算，但獲得的利潤預測僅為後者的 52%。

**給 MarketFirst 的策略建議**：考慮將行銷預算的 15–20%（約 $50,000–$70,000）重新分配至研發。根據模型估算，這樣的調整預期能提升利潤約 $38,250–$53,550（來自研發增加的貢獻），扣除行銷減少的損失（約 $1,450–$2,030），淨利潤提升約 $36,000–$52,000。

**案例 C：BalancedCorp（均衡型，含異常值特徵）**

BalancedCorp 是一間成熟的新創公司，採取均衡的支出策略。其年度支出結構如下：
- R&D Spend：$80,000（中等研發投入）
- Marketing Spend：$200,000（中等行銷投入）
- Administration：$130,000（正常的行政成本）

模型預測利潤：
```
Profit = 56,714 + 0.765 × 80,000 + 0.029 × 200,000
       = 56,714 + 61,200 + 5,800
       = $123,714
```

BalancedCorp 的預測利潤（$123,714）落在 TechGrowth 和 MarketFirst 之間。值得注意的是，BalancedCorp 的總支出（$410,000）約為 TechGrowth（$255,000）的 1.6 倍，但預測利潤僅為 TechGrowth 的 72%。這再次支持了「研發投入效率最高」的結論——在絕對支出上多投入不一定等於更高的利潤，支出的結構（研發 vs. 行銷 vs. 行政）才是關鍵。

**三案例對比總結**：

| 公司 | R&D | Marketing | 總支出 | 預測利潤 | 利潤/支出比 |
|------|-----|-----------|--------|----------|:---:|
| TechGrowth | $150K | $25K | $255K | $172,189 | 67.5% |
| BalancedCorp | $80K | $200K | $410K | $123,714 | 30.2% |
| MarketFirst | $30K | $350K | $500K | $89,814 | 18.0% |

利潤/支出比（Profit-to-Expense Ratio）的趨勢非常明顯：研發支出佔比越高，每一塊錢總支出產生的利潤就越高。這個簡單的案例研究清楚地展示了資料驅動的預算分配決策如何直接影響公司的盈利表現。

### 11.6 從數據到行動：五步驟決策流程

以下是一個五步驟的決策框架，幫助新創公司管理者將本分析的結論轉化為具體的行動計劃：

**步驟 1：診斷當前支出結構**
計算你的公司在 R&D、Marketing 和 Administration 上的支出比例。將這些比例與本資料集中表現最佳的公司的支出結構進行比較（高研發佔比、中等行銷、基本行政）。

**步驟 2：進行利潤預測**
使用最終迴歸方程式（Profit = 56,714 + 0.765 × R&D + 0.029 × Marketing）計算你的當前預測利潤。這提供了一個基準線，用於比較不同預算方案的預期效果。

**步驟 3：模擬預算重分配**
建立多個預算情境（Scenarios）：
- 情境 A：維持總預算不變，將 10% 行政支出轉移至研發
- 情境 B：維持總預算不變，將 20% 行銷支出轉移至研發
- 情境 C：增加總預算 15%，全部投入研發

對每個情境重新計算預測利潤，識別能最大化利潤的預算配置。

**步驟 4：加入現實約束條件**
模型的建議可能不完全符合業務現實。加入以下約束條件：
- 最低行銷支出門檻（維持品牌知名度所需）
- 人力資源限制（研發團隊擴張需要時間）
- 現金流限制（預算調整的幅度不能超過現金儲備的承受範圍）

**步驟 5：實施、監測、迭代**
執行選定的預算調整方案，並在 6–12 個月後評估實際結果：
- 比較實際利潤與模型預測利潤的差異
- 如果差異超過 ±RMSE（$5,450），分析原因（市場變化？執行問題？）
- 根據實際結果重新校準預算分配，形成持續優化的閉環

---

## 12. 常見問題與疑難排解（FAQ）

### 12.1 方法論相關問題

**Q1：為什麼同時使用五種特徵選取方法？只用一種不夠嗎？**

A1：不同的特徵選取方法基於不同的假設和機制。例如，Backward Elimination 依賴於 P-value（假設殘差常態性），而 Mutual Information 不依賴任何分佈假設。透過多方法交叉驗證，我們可以：
- 識別所有方法一致認可的「鐵證」特徵（如 R&D Spend 獲得 5/5 票）
- 發現方法之間的差異（如 Marketing Spend 在統計檢定中不顯著但在 MI 中排名第二）
- 對結論建立更高的信心——如果某個特徵只在單一方法中被選取，它很可能是一個虛假訊號

**Q2：為什麼 Backward Elimination 只選了一個特徵？**

A2：Backward Elimination 以 P-value = 0.05 作為門檻。在這個嚴格的標準下，Marketing Spend 的 P-value 略高於 0.05，因此被淘汰。這反映了 P-value 方法的保守性——它寧可遺漏一個邊際有用的特徵，也不願包含一個可能不顯著的特徵。如果在商業應用中對遺漏有用特徵的成本較高，可以考慮調高 α 至 0.10 或 0.15。

**Q3：為什麼 Lasso 保留了所有 5 個特徵，但性能最差？**

A3：Lasso 透過交叉驗證選擇 α（正則化強度）。當最優 α 值很小時，懲罰力度不足，Lasso 會保留所有特徵（趨近於普通 OLS）。全模型（5 特徵）的過擬合導致測試集性能下降。將門檻從 |coef| > 10⁻⁵ 提高到 |coef| > 0.01 可能會獲得更精簡的特徵集。

**Q4：Adjusted R² 和 R² 應該用哪個來比較不同特徵數的模型？**

A4：永遠使用 Adjusted R² 來比較不同特徵數量的模型。R² 永遠不會因增加特徵而下降（即使新增的是純隨機雜訊），因此使用 R² 來比較不同複雜度的模型會產生誤導。Adjusted R² 引入了對特徵數量的懲罰，只有在新增特徵的貢獻超過其「複雜度成本」時，Adjusted R² 才會提升。

### 12.2 資料相關問題

**Q5：為什麼不移除 R&D = 0 或 Marketing = 0 的公司？**

A5：這些公司雖然在特定支出上為零，但它們是真實存在的商業案例（有些新創公司確實沒有研發部門或不行銷）。在沒有明確證據表明這些是「資料錯誤」的情況下，不應隨意刪除觀測值。我們透過精煉模型（Model C: Huber 穩健迴歸）來評估這些潛在異常值的影響，而不是直接刪除它們。

**Q6：為什麼只使用 50 筆資料就能做分析？**

A6：50 筆資料對於線性迴歸來說確實偏少（自由度僅有 50 − p − 1，其中 p 是特徵數）。然而：
- 我們的主要目標是展示方法論而非建立一個生產級預測系統
- 特徵數量少（最多 5 個），樣本/特徵比為 10:1，勉強足夠
- 透過 Adjusted R² 和交叉驗證，我們已經考量了小樣本可能帶來的過度樂觀
- 本專案可作為更大規模資料集分析的藍本和模板

**Q7：資料集成為不平衡嗎？三個州的樣本數如何？**

A7：三個州的樣本數幾乎相等（California: 17, New York: 17, Florida: 16），不存在顯著的類別不平衡問題。此外，由於州變數最終被特徵選取方法認定為不顯著並被排除，類別分佈的輕微差異對最終模型沒有影響。

### 12.3 模型部署相關問題

**Q8：如何將模型部署到生產環境？**

A8：生產部署可以透過以下幾種方式：
1. **Streamlit Cloud**（最簡單）：本專案的 `app.py` 已經可以在 Streamlit Cloud 上執行，只需連接 GitHub 倉庫即可
2. **Flask/FastAPI API**：將最終的迴歸方程式實作為一個 REST API 端點，接受 R&D Spend 和 Marketing Spend 的 JSON 輸入，返回預測利潤
3. **Excel 試算表**：將方程式 `Profit = 56,714 + 0.765 × R&D + 0.029 × Marketing` 輸入 Excel，即可讓非技術使用者進行快速估算
4. **嵌入式系統**：方程式僅涉及兩個乘法與兩個加法運算，可輕易嵌入任何程式語言或系統中

**Q9：Streamlit Cloud 部署需要付費嗎？**

A9：Streamlit Cloud 提供免費層級（Free Tier），允許部署公開的 GitHub 倉庫應用程式。免費層級的限制包括：
- 應用程式在閒置一段時間後會進入休眠（再次訪問時自動喚醒）
- 資源配額有限（但對於本專案的輕量級應用來說完全足夠）
- 僅支援公開倉庫（私人倉庫需要付費方案）

對於個人專案與展示用途，免費層級完全滿足需求。

**Q10：GitHub Pages 儀表板如何更新？**

A10：GitHub Pages 儀表板的更新是全自動的。每次你推送（push）新的變更到 `master` 分支時，`.github/workflows/deploy.yml` 中定義的 GitHub Actions 工作流程會自動觸發，在 30–60 秒內完成重新部署。你不需要手動進行任何操作，只需確保 `index.html` 的變更已提交並推送。

### 12.4 擴展與客製化問題

**Q11：如何將這個分析框架應用到我的資料集？**

A11：遵循以下步驟：
1. 將你的 CSV 檔案放入 `data/raw/`
2. 修改 `main_analysis.py` 中的資料路徑和目標變數名稱
3. 如果你的資料包含類別變數（如同本專案的 State），確保在 One-Hot Encoding 步驟中正確指定
4. 根據需要調整顯著水準 α（更嚴格的 α 會保留更少的特徵）
5. 如果你的資料集較大（>1,000 筆），可以考慮將 TEST_SIZE 調小至 0.1

**Q12：可以加入非線性特徵或多項式項嗎？**

A12：本專案的基本框架可以擴展以支援非線性關係。建議的方法：
1. 使用 `PolynomialFeatures` 從原始特徵產生二次項和交互項
2. 將產生的多項式特徵加入設計矩陣
3. 重新執行特徵選取，檢查非線性項是否顯著
4. 注意：加入多項式項會大幅增加特徵數量，更容易過擬合——必須使用 Adjusted R² 和交叉驗證嚴格評估

**Q13：在 50 Startups 資料集上，是否能使用其他機器學習模型（如 Random Forest、XGBoost）？**

A13：當然可以。雖然本專案專注於線性迴歸（因為其高度的可解釋性），但你可以直接將相同的資料集和特徵用於其他模型。以下是一個簡單的比較框架：

1. **線性迴歸**（本專案使用）：R² ≈ 0.9474，最大優點是可解釋性（每個係數有明確的商業含義）
2. **Random Forest**：通常能達到更高的 R²（可能 0.96–0.98），但犧牲了可解釋性（無法得到簡單的方程式）
3. **XGBoost / LightGBM**：梯度提升樹通常在小資料上容易過擬合，但對於大資料集（>1,000 筆）可能是最佳選擇
4. **SVR（Support Vector Regression）**：對於含有異常值的資料較為穩健，但超參數調校較為複雜

實作建議：使用 scikit-learn 的 `Pipeline` 結合 `StandardScaler` 和所選模型，然後使用 `cross_val_score` 比較不同模型的 CV R²。

**Q14：如何評估模型在新資料上的表現？如何進行 A/B 測試？**

A14：在將模型部署到生產環境之前，建議進行以下驗證步驟：

1. **時間驗證（Temporal Validation）**：如果資料有時間戳記，使用較早的資料做訓練、較新的資料做測試。這比隨機分割更能反映真實世界中的模型表現。
2. **外部驗證（External Validation）**：在完全獨立的資料集上測試模型。即使資料來源不同（如不同國家的新創公司資料），如果模型仍能保持合理的預測準確度，說明模型具有良好的泛化能力。
3. **A/B 測試框架**：
   - A 組（控制組）：使用傳統方法（或人類專家）進行利潤預測
   - B 組（實驗組）：使用本模型進行利潤預測
   - 比較兩組的預測誤差（MAE 或 RMSE）以量化模型的實際效益
   - 記錄預測誤差的改善幅度，作為模型價值的量化證明

4. **持續監控**：部署後定期重新計算 RMSE 和 R²，如果性能顯著下降（例如 R² 下降超過 0.05），表示資料分佈可能發生了變化（概念漂移，Concept Drift），需要重新訓練模型。

**Q15：本專案的程式碼可以直接用於商業用途嗎？**

A15：是的，本專案使用 MIT 授權條款，這意味著：
- 你可以自由使用、修改和散佈程式碼
- 你可以將程式碼用於商業產品或服務
- 唯一的條件是保留原始的 MIT 授權聲明和版權標示
- 本軟體按「現狀」提供，作者不對使用結果提供任何保證

對於商業用途，建議在使用前進行以下調整：
1. 使用你自己的資料集進行完整的訓練和驗證
2. 進行更嚴格的異常值檢測和處理
3. 考慮使用自動化的超參數調校（如 `GridSearchCV`）
4. 增加模型的版本控制和變更追蹤機制
5. 建立自動化的模型監控和警報系統

### 12.5 延伸討論：方法論的批判性反思與未來方向

在結束本白皮書之前，我們對所使用的方法論進行一次批判性的反思，同時展望可能的改進方向。科學精神的核心在於對自身方法的持續質疑與改進。

**反思 1：線性模型的固有局限**

線性迴歸模型之所以被廣泛使用，是因為其高度的可解釋性——每個係數都有明確的「邊際效應」解釋。然而，現實世界中許多關係並非線性的。例如，R&D 支出對利潤的影響可能存在「報酬遞減」（Diminishing Returns）——最初的研發投入產生巨大的回報，但隨著支出增加到一定程度，每一塊錢新增支出的邊際利潤貢獻逐漸下降。

檢測非線性關係的方法包括：
- 在散佈圖中加入 LOWESS（Locally Weighted Scatterplot Smoothing）平滑曲線，觀察是否存在系統性的偏離直線的趨勢
- 使用多項式迴歸（Polynomial Regression）並比較 Adjusted R²，判斷加入二次項或三次項是否有實質改善
- 應用廣義加法模型（GAM, Generalized Additive Model），允許每個特徵具有靈活的非線性轉換

**反思 2：P-value 為基礎方法的爭議**

近年來，統計學界對 P-value 的使用展開了激烈的辯論。主要批評包括：
- **P-hacking**：研究者可能有意或無意地測試多個模型直到獲得 P < 0.05，然後僅報告「顯著」的結果
- **P-value 閾值的武斷性**：α = 0.05 是一個歷史慣例，而非基於任何數學或實務理由的絕對標準
- **P-value 不等於效果大小**：一個非常小的效果在大樣本中可能產生極小的 P-value，但該效果在實務上可能完全無關緊要

針對這些問題，本專案的應對策略包括：
- 使用五種方法交叉驗證（而非僅依賴 P-value）
- 同時報告效果大小（R²、RMSE、係數值）和統計顯著性（P-value）
- 使用 Adjusted R² 和交叉驗證來避免過度樂觀的評估

**反思 3：樣本量的限制與貝氏方法的潛力**

50 筆資料對於統計推論而言確實偏少。在小樣本的情境下，貝氏方法（Bayesian Methods）可能比頻率學派方法（Frequentist Methods，如 P-value）更為合適。貝氏線性迴歸可以：
- 透過先驗分佈（Prior Distribution）將領域知識正式納入模型
- 產生完整的後驗分佈（Posterior Distribution），而非單一的點估計——這提供了更豐富的不確定性量化
- 在小樣本時自動收縮估計值（透過先驗的規範化效果），避免過度擬合

使用 PyMC 或 Stan 等機率程式設計框架可以實現貝氏線性迴歸。對於本資料集，一個無資訊先驗（Uninformative Prior）的貝氏模型應該會給出與 OLS 非常相似的點估計，但會提供更真實的不確定性區間（特別是對於係數的標準誤）。

**反思 4：特徵工程的可能性**

本專案僅使用了原始的四個特徵（三個數值 + 一個類別）。但在實務中，特徵工程（Feature Engineering）往往能顯著提升模型性能。以下是一些潛在的特徵工程方向：

1. **交互作用項（Interaction Terms）**：
   - R&D × Marketing：研發和行銷之間可能存在協同效應——同時在兩方面都投入的公司可能獲得超過線性疊加的利潤
   - R&D × State：不同州的研發生態系統不同（如矽谷 vs. 其他地方），交互項可以捕捉這種差異

2. **比率變數（Ratio Variables）**：
   - R&D / Total_Expense：研發支出佔總支出的比例，反映公司的「技術導向程度」
   - Marketing / R&D：行銷與研發的支出比，反映公司的「市場導向 vs. 技術導向」策略定位

3. **基於領域知識的二元旗標**：
   - Is_Tech_Hub：公司所在地是否為科技中心（如加州為 1、其他為 0）
   - High_RD_Flag：研發支出是否高於中位數（可用於分段線性迴歸）

這些特徵工程雖然可能改善預測性能，但代價是增加了模型的複雜度和解釋難度。在加入任何衍生特徵後，應使用交叉驗證驗證其是否真正提升了泛化能力。

**反思 5：從預測模型到因果推論的鴻溝**

本專案建立的是**預測模型**（Predictive Model），而非**因果模型**（Causal Model）。這是一個關鍵的區別：
- 預測模型回答：「給定觀測到的支出結構，預期的利潤是多少？」
- 因果模型回答：「如果我們**主動改變**研發支出，利潤會如何**變化**？」

從預測到因果的轉變需要額外的假設和方法：
- **工具變數（Instrumental Variables）**：找到一個影響 R&D 支出但不直接影響利潤的外部變數（如政府研發補貼的資格門檻）
- **DID（Difference-in-Differences）**：如果有前後兩個時期的資料，可以比較政策改變前後的差異
- **RDD（Regression Discontinuity Design）**：利用政策門檻的自然實驗（如研發稅務優惠的收入門檻）

沒有這些方法，我們不能簡單地斷言「增加 R&D 支出就能增加利潤」——可能存在未觀測的混淆因子（如公司管理品質）同時影響了 R&D 支出和利潤。

**未來方向**：

基於以上反思，以下是本專案可能的未來發展方向：

1. **多時期面板資料分析**：若能獲得同一批公司多年的追蹤資料，可以使用固定效應模型（Fixed Effects Model）控制公司層級未觀測的恆定特徵，獲得更接近因果關係的估計。

2. **機器學習模型的比較基準**：系統性地比較線性迴歸與 Random Forest、Gradient Boosting、Neural Network 在相同資料集上的表現，量化「可解釋性」與「預測準確度」之間的權衡。

3. **自動化超參數調校**：使用 `GridSearchCV` 或 `RandomizedSearchCV` 自動探索 Lasso α、RFE 特徵數、Huber δ 等超參數的最優組合，進一步提升模型性能。

4. **互動式決策支援系統**：將最終模型嵌入一個可以讓非技術使用者輸入支出預算並即時看到預測利潤的 Web 應用程式（Streamlit 儀表板的進化版），包含情境分析（What-If Analysis）和敏感度分析功能。

5. **模型公平性與倫理評估**：如果模型被用於影響資源分配決策（如投資決策），需要評估模型是否對不同類型的公司（如不同規模、不同產業）存在系統性的預測偏差。

### 12.6 學習資源與進階路徑

對於希望深入學習本白皮書所涉及主題的讀者，以下是一條建議的學習路徑：

**初階（1–2 週）**：
- 完成本專案的 Quick Start（第 10 章），實際執行所有腳本
- 閱讀 James et al. 的 *ISLR* 第 3 章（Linear Regression），搭配本專案的 50 Startups 資料進行實作練習
- 使用 Jupyter Notebook 逐 Cell 重現 main_analysis.py 的邏輯，加深對每一步的理解

**中階（2–4 週）**：
- 深入研究五種特徵選取方法的理論基礎（參考文獻 3–5）
- 在自己的資料集上應用本專案的框架，比較不同方法的選取結果
- 學習使用 `statsmodels` 進行完整的迴歸診斷（殘差分析、影響點檢測、共線性評估）
- 嘗試特徵工程（加入交互項、比率變數），觀察對模型的影響

**進階（1–2 個月）**：
- 學習貝氏線性迴歸（使用 PyMC 或 Stan），比較頻率學派與貝氏方法的結果
- 將線性模型與非線性模型（Random Forest、XGBoost）進行系統性比較
- 研究因果推論方法（工具變數、DID、RDD），理解預測與因果的區別
- 將模型部署為生產級 API（使用 FastAPI + Docker），學習 MLOps 基礎



---

## 13. 附錄

### 13.1 產出圖表清單

以下為 `outputs/figures/` 目錄中自動生成的 11 張 PNG 圖表及其技術規格：

| 檔名 | 描述 | 解析度 | 格式 |
|------|------|--------|------|
| `corr_heatmap.png` | Pearson 相關性熱力圖（含註解數值），使用上三角遮罩 | 150 dpi | PNG |
| `scatter_features.png` | R&D、Admin、Marketing 三者與 Profit 的並排散佈圖，含相關係數標註 | 150 dpi | PNG |
| `boxplot_state.png` | 三個州的 Profit 箱型圖 + 蜂群疊加圖（stripplot） | 150 dpi | PNG |
| `feature_count_performance.png` | 特徵數（1→5）vs RMSE 和 R² 的雙軸折線圖 | 150 dpi | PNG |
| `feature_selection_comparison.png` | 5 方法特徵選取熱力圖（方法 × 特徵）+ R² 水平長條圖 | 150 dpi | PNG |
| `method_comparison_rmse_r2.png` | 5 方法的 RMSE 和 R² 水平長條圖（並排比較） | 150 dpi | PNG |
| `method_comparison_heatmap_performance.png` | 特徵選取熱力圖 + 特徵數 vs 性能散佈圖（以方法著色） | 150 dpi | PNG |
| `method_comparison_dashboard.png` | 4 合 1 全覽儀表板（RMSE 圖、R² 圖、熱力圖、摘要表） | 150 dpi | PNG |
| `model_comparison.png` | 原始模型 vs 精煉模型 A/B/C 的 R²/MAE/RMSE 分組長條圖 | 150 dpi | PNG |
| `pred_vs_actual.png` | 測試集預測值 vs 實際值散佈圖，含完美預測線（y=x）和 R²/Adj.R²/RMSE 標註 | 150 dpi | PNG |
| `residuals.png` | 雙面板殘差圖：左為殘差 vs 預測值散佈圖，右為殘差直方圖（附常態參考線） | 150 dpi | PNG |

### 13.2 報告清單

以下為 `outputs/reports/` 目錄中自動生成的 6 份報告文件：

| 檔名 | 格式 | 內容描述 |
|------|------|------|
| `metrics.txt` | 純文字 | 基礎 OLS 模型的 statsmodels 完整摘要（係數表、P-value、R²、F 檢定、AIC/BIC、診斷指標） |
| `refined_models.txt` | 純文字 | Model A/B/C 三種精煉策略的詳細比較報告（含方程式、評估指標、診斷結果） |
| `feature_selection.txt` | 純文字 | 5 種特徵選取方法的詳細報告（每種方法的選取過程、最終特徵、投票結果） |
| `feature_selection_results.csv` | CSV | 循序特徵新增分析數據（特徵數、特徵組合、RMSE、R²） |
| `feature_selection_results.tsv` | TSV | 同上，使用 Tab 分隔（可在 Excel 中直接開啟而不會出現編碼問題） |
| `method_comparison_results.csv` | CSV | 5 方法比較數據（方法名稱、特徵數、選取特徵、R²、Adj.R²、RMSE、CV R² Mean、CV R² Std） |

### 13.3 詞彙表

| 術語 | 英文 | 定義 |
|------|------|------|
| 虛擬變數陷阱 | Dummy Variable Trap | 當 k 個虛擬變數全部保留時產生的完全共線性問題，使得正規方程式不可逆。解決方案為刪除其中一個變數（k-1 編碼）。 |
| 逐步淘汰法 | Backward Elimination | 從全模型開始，逐步移除 P-value 最高且超過顯著水準 α 的不顯著變數，直到所有保留變數均顯著為止。 |
| 逐步選擇法 | Forward Selection | 從空模型開始，逐步加入 P-value 最低且低於顯著水準 α 的顯著變數，直到無變數可加入為止。 |
| 遞迴特徵消除 | RFE（Recursive Feature Elimination） | 迭代訓練模型並淘汰權重最小的特徵，直到達到指定的特徵數量。 |
| L1 正則化 | Lasso（Least Absolute Shrinkage and Selection Operator） | 在損失函數中加入 L1 懲罰項 α × Σ|β|，使部分係數恰好歸零，同時實現特徵選取與正則化。 |
| 互信息 | Mutual Information | 資訊理論中的非線性相依度量，量化知道一個變數能減少多少關於另一個變數的不確定性。 |
| 變異數膨脹因子 | VIF（Variance Inflation Factor） | 衡量多重共線性程度的指標。VIFⱼ = 1/(1−R²ⱼ)，其中 R²ⱼ 是以 Xⱼ 為目標、其餘特徵為預測變數的 R²。VIF > 10 表示嚴重共線性。 |
| Cook's 距離 | Cook's Distance | 衡量單一資料點對整個迴歸模型影響力的指標。Dᵢ > 4/n 被視為高影響點，需進一步檢視。 |
| Box-Cox 轉換 | Box-Cox Transformation | 對正數變數應用的冪次轉換 y(λ) = (y^λ − 1)/λ，目的為使轉換後的變數更接近常態分佈。 |
| Omnibus 檢定 | Omnibus Test | 綜合偏態與峰度的殘差常態性檢定。虛無假設為殘差服從常態分佈，p < 0.05 表示拒絕常態性假設。 |
| Durbin-Watson 檢定 | Durbin-Watson Test | 檢測殘差一階自相關的統計量。DW ≈ 2 表示無自相關，DW < 1 或 > 3 表示顯著自相關。 |
| 均方根誤差 | RMSE（Root Mean Squared Error） | √(Σ(ŷ−y)²/n)，對大誤差施以較重懲罰的預測誤差度量。 |
| 交叉驗證 | Cross-Validation | 將資料分為 k 個折疊（fold），輪流以其中一個折疊作為驗證集、其餘作為訓練集，重複 k 次取平均績效，以評估模型的泛化能力。 |

### 13.4 參考文獻與延伸閱讀

以下資源提供了本白皮書所涉及方法論的更深層次討論：

1. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
2. Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). *Introduction to Linear Regression Analysis* (5th ed.). Wiley.
3. Guyon, I., & Elisseeff, A. (2003). An Introduction to Variable and Feature Selection. *Journal of Machine Learning Research*, 3, 1157–1182.
4. Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.
5. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.
6. Sheather, S. (2009). *A Modern Approach to Regression with R*. Springer.
7. Streamlit Documentation. (2024). https://docs.streamlit.io/
8. Plotly Python Graphing Library. (2024). https://plotly.com/python/

---

## 14. 變更日誌 (Changelog)

| 版本 | 日期 | 變更內容 |
|------|------|---------|
| 1.0 | 2026-06-09 | 初始版本：完整 ML 管線、5 種特徵選取方法、11 張圖表、6 份報告 |
| 1.1 | 2026-06-09 | 加入 GitHub Pages 儀表板（`index.html`）與 CI/CD 部署管線（`.github/workflows/deploy.yml`） |
| 1.2 | 2026-06-09 | 加入 `WHITEPAPER.md` 技術白皮書（10 章節） |
| 1.3 | 2026-06-09 | 加入 `README.md`（含 Badge 與 Live Demo 連結）及 `.gitignore`（排除 .env） |
| 1.4 | 2026-06-09 | 加入 `design.md` 架構設計文件、`supplement_analysis.py` 補充診斷腳本 |
| 1.5 | 2026-06-09 | 儀表板新增：Method Comparison Results 表格、Sequential Addition 表格、Feature Votes 卡片 |
| 1.6 | 2026-06-09 | 封裝 `ml-regression-pipeline` OpenCode Skill（16 步驟完整工作流程） |
| 1.7 | 2026-06-09 | 白皮書更新：第 9 章部署架構擴充儀表板區塊詳情、第 11 章新增變更日誌 |
| 1.8 | 2026-06-11 | 加入 `app.py` Streamlit CRISP-DM 互動式儀表板（4 章節、雙線聯動圖、hover 特徵組合）；技術棧新增 Streamlit ≥ 1.55 與 Plotly ≥ 5.18；部署架構新增 Streamlit Cloud 部署（[l6-new-model.streamlit.app](https://l6-new-model.streamlit.app/)）；README 與 requirements.txt 同步更新 |
| 1.9 | 2026-06-11 | 白皮書大幅擴充（從 ~2,600 字擴增至 20,000+ 字）：每章節新增詳細的教學級內容，包括 CRISP-DM 六階段完整說明、EDA 指南、前處理最佳實踐、五種特徵選取方法的數學原理與程式碼、評估指標詳解、高斯-馬可夫假設檢驗、參考文獻 |
| 2.0 | 2026-06-11 | 新增第 10–12 章：快速入門與實作教學（含環境設置、腳本執行流程、Streamlit 操作指南、自訂參數說明、常見問題解決方案）、商業解讀與決策指南（含核心商業洞察、投資決策框架、模型限制、利害關係人溝通要點）、FAQ 疑難排解（含方法論、資料、部署、擴展四大類共 12 題）|
| 2.1 | 2026-06-11 | 第 5 章新增 Backward Elimination 逐輪實戰演練與方法選擇建議；第 8 章新增診斷失敗應對策略與 10 項全面檢核表；第 10 章新增 Docker 容器化部署、Jupyter Notebook 教學、輸出解讀指南、專案維護與跨領域應用指南；第 11 章新增三間虛構公司案例研究與五步驟決策流程；第 12 章新增 3 題 FAQ、方法論批判性反思與未來方向、學習資源與進階路徑；第 3 章新增相關係數深入解讀與視覺化最佳實踐 |
| 2.2 | 2026-06-11 | 儀表板全面升級為手繪 Excalidraw 風格（`infographic.html` / `index.html`）：移除 Tailwind CSS 依賴改為純手繪 CSS、新增雙頁籤切換（Interactive Infographic / Static Poster PNG）、互動式 SVG SFA 圖（R²/RMSE 切換按鈕 + hover tooltip）、手繪 CRISP-DM 工作流程圖、線框 Streamlit 儀表板預覽、`ml_pipeline_infographic.png` 靜態海報；第 9 章部署架構同步更新 |

---

> **文件目的**：本白皮書作為 L6 Crisp-RD2 專案的技術規格文件，詳載了系統架構、方法論、實驗結果、部署細節（含 Streamlit Cloud 與 GitHub Pages 雙部署），以及完整的實作教學、商業解讀指南與 FAQ 疑難排解。任何對專案的修改應先參照本文，確保一致性與可重現性。本文亦設計為教學級文件，適合資料科學學習者作為多元線性迴歸專案的參考教材。
