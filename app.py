import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRISP-DM Startup Profit Dashboard",
    page_icon="📊",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────
# Hardcoded raw data (50 startups)
# ──────────────────────────────────────────────────────────────────────
RAW_ROWS = [
    [165349.20, 136897.80, 471784.10, "New York", 192261.83],
    [162597.70, 151377.59, 443898.53, "California", 191792.06],
    [153441.51, 101145.55, 407934.54, "Florida", 191050.39],
    [144372.41, 118671.85, 383199.62, "New York", 182901.99],
    [142107.34, 91391.77, 366168.42, "Florida", 166187.94],
    [131876.90, 99814.71, 362861.36, "New York", 156991.12],
    [134615.46, 147198.87, 127716.82, "California", 156122.51],
    [130298.13, 145530.06, 323876.68, "Florida", 155752.60],
    [120542.52, 148718.95, 311613.29, "New York", 152211.77],
    [123334.88, 108679.17, 304981.62, "California", 149759.96],
    [101913.08, 110594.11, 229160.95, "Florida", 146121.95],
    [100671.96, 91790.61, 249744.55, "California", 144259.40],
    [93863.75, 127320.38, 249839.44, "Florida", 141585.52],
    [91992.39, 135495.07, 252664.93, "California", 134307.35],
    [119943.24, 156547.42, 256512.92, "Florida", 132602.65],
    [114523.61, 122616.84, 261776.23, "New York", 129917.04],
    [78013.11, 121597.55, 264346.06, "California", 126992.93],
    [94657.16, 145077.58, 282574.31, "New York", 125370.37],
    [91749.16, 114175.79, 294919.57, "Florida", 124266.90],
    [86419.70, 153514.11, 0.00, "New York", 122776.86],
    [76253.86, 113867.30, 298664.47, "California", 118474.03],
    [78389.47, 153773.43, 299737.29, "New York", 111313.02],
    [73994.56, 122782.75, 303319.26, "Florida", 110352.25],
    [67532.53, 105751.03, 304768.73, "Florida", 108733.99],
    [77044.01, 99281.34, 140574.81, "New York", 108552.04],
    [64664.71, 139553.16, 137962.62, "California", 107404.34],
    [75328.87, 144135.98, 134050.07, "Florida", 105733.54],
    [72107.60, 127864.55, 353183.81, "New York", 105008.31],
    [66051.52, 182645.56, 118148.20, "Florida", 103282.38],
    [65605.48, 153032.06, 107138.38, "New York", 101004.64],
    [61994.48, 115641.28, 91131.24, "Florida", 99937.59],
    [61136.38, 152701.92, 88218.23, "New York", 97483.56],
    [63408.86, 129219.61, 46085.25, "California", 97427.84],
    [55493.95, 103057.49, 214634.81, "Florida", 96778.92],
    [46426.07, 157693.92, 210797.67, "California", 96712.80],
    [46014.02, 85047.44, 205517.64, "New York", 96479.51],
    [28663.76, 127056.21, 201126.82, "Florida", 90708.19],
    [44069.95, 51283.14, 197029.42, "California", 89949.14],
    [20229.59, 65947.93, 185265.10, "New York", 81229.06],
    [38558.51, 82982.09, 174999.30, "California", 81005.76],
    [28754.33, 118546.05, 172795.67, "California", 78239.91],
    [27892.92, 84710.77, 164470.71, "Florida", 77798.83],
    [23640.93, 96189.63, 148001.11, "California", 71498.49],
    [15505.73, 127382.30, 35534.17, "New York", 69758.98],
    [22177.74, 154806.14, 28334.72, "California", 65200.33],
    [1000.23, 124153.04, 1903.93, "New York", 64926.08],
    [1315.46, 115816.21, 297114.46, "Florida", 49490.75],
    [0.00, 135426.92, 0.00, "California", 42559.73],
    [542.05, 51743.15, 0.00, "New York", 35673.41],
    [0.00, 116983.80, 45173.06, "California", 14681.40],
]
COLUMNS = ["R&D Spend", "Administration", "Marketing Spend", "State", "Profit"]
df_raw = pd.DataFrame(RAW_ROWS, columns=COLUMNS)

# ──────────────────────────────────────────────────────────────────────
# Hardcoded Sequential Feature Addition results
# ──────────────────────────────────────────────────────────────────────
SFA_DATA = {
    "Number of Features": [1, 2, 3, 4, 5],
    "Selected Features": [
        "R&D Spend",
        "R&D Spend, Marketing Spend",
        "R&D Spend, Marketing Spend, State_Florida",
        "R&D Spend, Marketing Spend, State_Florida, Administration",
        "R&D Spend, Marketing Spend, State_Florida, Administration, State_New York",
    ],
    "RMSE": [8274.87, 8198.80, 8376.45, 9068.54, 9137.99],
    "R-squared": [0.9465, 0.9474, 0.9451, 0.9357, 0.9347],
}
df_sfa = pd.DataFrame(SFA_DATA)

# Best model index
BEST_IDX = df_sfa["RMSE"].idxmin()
BEST_N = df_sfa.loc[BEST_IDX, "Number of Features"]
BEST_RMSE = df_sfa.loc[BEST_IDX, "RMSE"]
BEST_R2 = df_sfa.loc[BEST_IDX, "R-squared"]
BEST_FEATURES = df_sfa.loc[BEST_IDX, "Selected Features"]

# ──────────────────────────────────────────────────────────────────────
# Custom CSS for section divider styling
# ──────────────────────────────────────────────────────────────────────
SECTION_PREFIX = "## :gray[━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━]"

# ══════════════════════════════════════════════════════════════════════
# CHAPTER 1 — Project Overview
# ══════════════════════════════════════════════════════════════════════
st.markdown(SECTION_PREFIX)
st.title("📊 CRISP-DM Startup Profit Prediction Dashboard")
st.caption("**L6 Crisp-RD2** · Predicting Profit for 50 Startups using Multiple Linear Regression")

col1, col2 = st.columns([3, 1])
with col1:
    st.info(
        "**🎯 Mission Statement**\n\n"
        "This project applies the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) "
        "methodology to build a Multiple Linear Regression model that accurately predicts "
        "a startup's **Profit** based on its R&D spending, Marketing expenditure, "
        "Administration costs, and geographic location (State).\n\n"
        "By systematically comparing **5 feature selection methods** — Backward Elimination, "
        "Forward Selection, RFE, Lasso Regression, and Mutual Information — we identify the "
        "optimal feature subset that maximises predictive performance while minimising complexity."
    )
with col2:
    st.metric("Dataset Size", f"{len(df_raw)}", "startups")
    st.metric("Features", "4", "+ State")
    st.metric("Target", "Profit", "$")

st.markdown(SECTION_PREFIX)

# ══════════════════════════════════════════════════════════════════════
# CHAPTER 2 — Data Discovery
# ══════════════════════════════════════════════════════════════════════
st.header("📋 Data Discovery")
st.markdown("Explore the raw 50 Startups dataset and its statistical properties.")

# ── Key metric cards ──────────────────────────────────────────────────
avg_rd = df_raw["R&D Spend"].mean()
avg_mkt = df_raw["Marketing Spend"].mean()
avg_profit = df_raw["Profit"].mean()

kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric("Avg R&D Spend", f"${avg_rd:,.2f}")
with kpi2:
    st.metric("Avg Marketing Spend", f"${avg_mkt:,.2f}")
with kpi3:
    st.metric("Avg Profit", f"${avg_profit:,.2f}")

# ── Statistical summary (always visible) ──────────────────────────────
st.markdown("#### Statistical Summary")
st.dataframe(
    df_raw.describe().style.format("{:,.2f}"),
    use_container_width=True,
)

# ── Raw data toggle ───────────────────────────────────────────────────
if st.checkbox("Show / Hide Raw Data (50 rows)", value=False):
    st.markdown("#### Raw 50 Startups Dataset")
    st.dataframe(
        df_raw.style.format({
            "R&D Spend": "${:,.2f}",
            "Administration": "${:,.2f}",
            "Marketing Spend": "${:,.2f}",
            "Profit": "${:,.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.markdown(SECTION_PREFIX)

# ══════════════════════════════════════════════════════════════════════
# CHAPTER 3 — CRISP-DM Modeling Workflow
# ══════════════════════════════════════════════════════════════════════
st.header("⚙️ CRISP-DM Modeling Workflow")
st.markdown("Follow the end-to-end pipeline from raw data to trained model.")

step1, step2, step3, step4 = st.tabs([
    "Step 1: Data Cleaning",
    "Step 2: Feature Encoding",
    "Step 3: Train-Test Split",
    "Step 4: MLR Modeling",
])

with step1:
    st.markdown("### Step 1 — Data Cleaning")
    st.markdown("""
    - **Missing Values**: The 50 Startups dataset contains **no missing values** across all columns.
    - **Outlier Detection**: We examined the distribution of each numeric feature. While a few
      companies have **zero Marketing Spend**, these are legitimate observations (some startups
      allocate $0 to marketing) and were retained.
    - **Result**: Dataset is clean and ready for preprocessing (50 rows × 5 columns).
    """)
    st.code(f"""# Missing value check
df_raw.isnull().sum()
# ── All columns: 0 missing values ──

# Data shape
df_raw.shape
# ── (50, 5) ──""", language="python")

with step2:
    st.markdown("### Step 2 — Feature Encoding")
    st.markdown("""
    The **State** column is categorical with 3 unique values: **New York**, **California**, **Florida**.
    - Used **One-Hot Encoding** (`pd.get_dummies`) to convert State into binary dummy variables.
    - Applied the **Dummy Variable Trap** avoidance rule: dropped the first dummy column
      (California) to serve as the baseline category.
    - Final encoded features: `State_Florida`, `State_New York` (California = 0 on both).
    - All numeric features (**R&D Spend**, **Administration**, **Marketing Spend**) were
      **standardised** using `StandardScaler`.
    """)
    st.code("""# One-Hot Encoding
state_dummies = pd.get_dummies(df["State"], prefix="State", dtype=int)
state_dummies = state_dummies.drop(columns=["State_California"])
# ── Baseline: California ──

# StandardScaler on numeric features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled[num_features] = scaler.fit_transform(X[num_features])""", language="python")

with step3:
    st.markdown("### Step 3 — Train-Test Split")
    st.markdown("""
    - **Split Ratio**: 80% training / 20% testing
    - **Random State**: 0 (for reproducibility)
    - **Training Set**: 40 samples
    - **Testing Set**: 10 samples
    """)
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Training Set", "40", "80%")
    with col_b:
        st.metric("Testing Set", "10", "20%")
    st.code("""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=0
)
# ── X_train: (40, 5)  X_test: (10, 5) ──""", language="python")

with step4:
    st.markdown("### Step 4 — Multiple Linear Regression Modeling")
    st.markdown("""
    - **Algorithm**: Ordinary Least Squares (OLS) Multiple Linear Regression
    - **Feature Selection**: Backward Elimination (P-value threshold = 0.05)
    - **Evaluation Metrics**: R-squared, Adjusted R-squared, RMSE, MAE
    - **Final Model**: Selected features after elimination — **R&D Spend** and **Marketing Spend**
      (other features were statistically insignificant at α = 0.05)
    """)
    st.code("""# statsmodels OLS with Backward Elimination
import statsmodels.api as sm
X_sm = sm.add_constant(X_train)
model = sm.OLS(y_train, X_sm).fit()

# Backward Elimination:
#  Round 1 — max P-value: State_New York (0.990) → REMOVED
#  Round 2 — max P-value: Administration (0.602) → REMOVED
#  Round 3 — max P-value: State_Florida (0.663) → REMOVED
#  Final: R&D Spend + Marketing Spend""", language="python")

st.markdown(SECTION_PREFIX)

# ══════════════════════════════════════════════════════════════════════
# CHAPTER 4 — Advanced Feature Selection Analysis
# ══════════════════════════════════════════════════════════════════════
st.header("🔬 Advanced Feature Selection Analysis")
st.markdown(
    "This section presents the **Sequential Feature Addition (SFA)** analysis — systematically "
    "adding features one by one and tracking how model performance (RMSE & R²) evolves."
)

# ── 4.1 Interactive Data Table ────────────────────────────────────────
st.subheader("Sequential Feature Addition — Results Table")
st.dataframe(
    df_sfa.style.format({"RMSE": "${:,.2f}", "R-squared": "{:.4f}"}),
    column_config={
        "Number of Features": st.column_config.NumberColumn("Number of Features"),
        "Selected Features": st.column_config.TextColumn("Selected Features"),
        "RMSE": st.column_config.NumberColumn("RMSE ($)", format="$%.2f"),
        "R-squared": st.column_config.NumberColumn("R²", format="%.4f"),
    },
    use_container_width=True,
    hide_index=True,
)

# ── 4.2 Interactive Toggle SFA Chart ──────────────────────────────────
st.subheader("Performance Trends — Toggle RMSE & R²")
st.caption(
    "💡 **Click the buttons below** to switch between RMSE and R². **Hover** over any data point "
    "to see the exact feature combination and metric value."
)

metric_toggle = st.radio(
    "Select Metric",
    options=["Test R² (Higher is Better)", "Test RMSE (Lower is Better)"],
    horizontal=True,
    label_visibility="collapsed",
)

if "R²" in metric_toggle:
    y_col = "R-squared"
    y_label = "R²"
    y_fmt = ".4f"
    line_color = "#10b981"
    best_y = BEST_R2
    best_text_pos = "bottom center"
    best_label = "(HIGHEST)"
    y_range = [0.92, 0.96]
    _title = "R-squared by Number of Features"
else:
    y_col = "RMSE"
    y_label = "RMSE ($)"
    y_fmt = ",.2f"
    line_color = "#f97316"
    best_y = BEST_RMSE
    best_text_pos = "top center"
    best_label = "(LOWEST)"
    y_range = None
    _title = "RMSE by Number of Features"

fig_toggle = px.line(
    df_sfa,
    x="Number of Features",
    y=y_col,
    markers=True,
    title=_title,
    labels={"Number of Features": "Number of Features", y_col: y_label},
    custom_data=["Selected Features"],
)
fig_toggle.update_traces(
    line=dict(color=line_color, width=3),
    marker=dict(size=14, color=line_color, line=dict(width=1, color="white")),
    hovertemplate=(
        "<b>Features</b>: %{customdata[0]}<br>"
        "<b>Feature Count</b>: %{x}<br>"
        f"<b>{y_label}</b>: %{{y:{y_fmt}}}<extra></extra>"
    ),
)
# Star marker for optimal point
fig_toggle.add_trace(
    go.Scatter(
        x=[BEST_N],
        y=[best_y],
        mode="markers+text",
        name="Optimal",
        marker=dict(size=20, color="#facc15", symbol="star", line=dict(width=2, color="#1e1e1e")),
        text=["▼ BEST"],
        textposition=best_text_pos,
        textfont=dict(size=12, color="#1e1e1e", family="Arial Black"),
        hovertemplate=(
            "<b>Features</b>: %{customdata[0]}<br>"
            "<b>Feature Count</b>: %{x}<br>"
            f"<b>{y_label}</b>: %{{y:{y_fmt}}} " + best_label + "<extra></extra>"
        ),
        customdata=[[BEST_FEATURES]],
    )
)
fig_toggle.update_layout(
    xaxis=dict(tickmode="linear", tick0=1, dtick=1, title="Number of Features"),
    yaxis=dict(title=y_label),
    showlegend=False,
    hovermode="closest",
)
if y_range:
    fig_toggle.update_layout(yaxis=dict(range=y_range, title=y_label))

st.plotly_chart(fig_toggle, use_container_width=True)

# ── 4.3 Insight Block ─────────────────────────────────────────────────
st.divider()
st.success(
    f"### 🏆 Optimal Model Found: {BEST_N} Features\n\n"
    f"**Feature Combination**: {BEST_FEATURES}\n\n"
    f"The model achieves its **best performance at the 2-feature mark** — this is the "
    f"**Elbow Point** of the analysis:\n\n"
    f"- **RMSE = ${BEST_RMSE:,.2f}** (lowest across all configurations)\n"
    f"- **R² = {BEST_R2:.4f}** (highest across all configurations)\n\n"
    f"**Why does performance degrade with more features?**\n\n"
    f"Adding `State_Florida`, `Administration`, and `State_New York` introduces noise "
    f"and unnecessary complexity. These features have **low correlation with Profit** and "
    f"fail to improve the model's predictive power. Instead, they cause **overfitting** — "
    f"the model learns spurious patterns in the training data that do not generalise to "
    f"unseen samples, resulting in **higher RMSE** and **lower R²**.\n\n"
    f"**Conclusion**: The most parsimonious and accurate model uses only **R&D Spend** "
    f"and **Marketing Spend** to predict Profit."
)

st.markdown(SECTION_PREFIX)
st.caption("📊 CRISP-DM Startup Profit Prediction Dashboard · Built with Streamlit + Plotly")
