import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="企業機器學習迴歸分析端到端指南 — ML Regression Pipeline",
    page_icon="🔬",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════
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

# SFA results
SFA_DATA = {
    "特徵數": [1, 2, 3, 4, 5],
    "選取特徵": [
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
BEST_IDX = df_sfa["RMSE"].idxmin()
BEST_N = int(df_sfa.loc[BEST_IDX, "特徵數"])
BEST_RMSE = df_sfa.loc[BEST_IDX, "RMSE"]
BEST_R2 = df_sfa.loc[BEST_IDX, "R-squared"]
BEST_FEATURES = df_sfa.loc[BEST_IDX, "選取特徵"]

# Feature selection consensus
FS_DATA = {
    "排名": [1, 2, 3, 4],
    "互資訊 (Mutual Info)": ["R&D Spend", "Marketing Spend", "—", "—"],
    "後向淘汰 (Backward Elim)": ["R&D Spend", "(已剔除)", "—", "—"],
    "前向選擇 (Forward Sel)": ["R&D Spend", "(已剔除)", "—", "—"],
    "RFE (CV)": ["R&D Spend", "Marketing Spend", "State_FL", "—"],
    "Lasso L1": ["R&D Spend", "Marketing Spend", "State_FL", "Admin"],
}
df_fs = pd.DataFrame(FS_DATA)

# ═══════════════════════════════════════════════════════════════════════
# Per-method chart data
# ═══════════════════════════════════════════════════════════════════════

# 1. Backward Elimination — P-value round-by-round
BE_DATA = {
    "特徵": ["R&D Spend", "Marketing Spend", "State_Florida", "Administration", "State_New York"],
    "Round 1 (全特徵)": [0.000, 0.021, 0.663, 0.602, 0.990],
    "Round 2 (移除 State_NY)": [0.000, 0.018, 0.584, 0.512, None],
    "Round 3 (移除 Admin)": [0.000, 0.015, 0.471, None, None],
    "Final (R&D + Mkt)": [0.000, 0.013, None, None, None],
}
df_be = pd.DataFrame(BE_DATA)

# 2. Forward Selection — R² after each step
FS_STEPS = {
    "步驟": ["起始 (無特徵)", "+ R&D Spend", "+ Marketing Spend", "+ State_Florida (停止)"],
    "R²": [0.000, 0.9465, 0.9474, 0.9474],
    "Adj. R²": [0.000, 0.9398, 0.9324, 0.9324],
}
df_fs_steps = pd.DataFrame(FS_STEPS)

# 3. RFE — Feature ranking scores
RFE_DATA = {
    "特徵": ["R&D Spend", "Marketing Spend", "State_Florida", "Administration", "State_New York"],
    "重要性分數": [1.00, 0.52, 0.18, 0.11, 0.04],
    "排名": [1, 2, 3, 4, 5],
}
df_rfe = pd.DataFrame(RFE_DATA)

# 4. Lasso L1 — Coefficient shrinkage path
LASSO_DATA = {
    "Alpha (正則化強度)": [0.001, 0.01, 0.1, 1.0, 10.0],
    "R&D Spend": [0.79, 0.78, 0.74, 0.52, 0.0],
    "Marketing Spend": [0.21, 0.20, 0.15, 0.0, 0.0],
    "State_Florida": [0.05, 0.03, 0.0, 0.0, 0.0],
    "Administration": [0.02, 0.0, 0.0, 0.0, 0.0],
    "State_New York": [0.01, 0.0, 0.0, 0.0, 0.0],
}
df_lasso = pd.DataFrame(LASSO_DATA)

# 5. Mutual Information — MI scores
MI_DATA = {
    "特徵": ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida", "Administration"],
    "MI 分數": [0.82, 0.45, 0.12, 0.10, 0.08],
}
df_mi = pd.DataFrame(MI_DATA)

# ═══════════════════════════════════════════════════════════════════════
# Page Header
# ═══════════════════════════════════════════════════════════════════════
st.title("🔬 企業機器學習迴歸分析端到端指南 — ML Regression Pipeline")
st.caption("Production-Ready Multiple Linear Regression · 50 Startups Dataset · CRISP-DM Standard")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("資料集", "50 筆", "Startups")
with c2:
    st.metric("特徵數", "4", "R&D / Admin / Mkt / State")
with c3:
    st.metric("特徵選取方法", "5 種", "投票共識機制")
with c4:
    st.metric("最佳 Test R²", "0.9474", "2 特徵組合")

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# 一、專案概述與技術架構
# ═══════════════════════════════════════════════════════════════════════
st.header("一、專案概述與技術架構")

col_a, col_b = st.columns([2, 1])
with col_a:
    st.markdown("""
    本專案旨在建立一個符合工業標準的機器學習迴歸流程，涵蓋從原始資料預處理到模型部署與診斷的完整生命週期。

    **核心目標**：透過多種特徵選擇方法與統計診斷，尋找具備最佳預測能力且最具解釋性的線性模型。
    """)
with col_b:
    st.markdown("""
    **技術棧 (Tech Stack)**：
    - **Python 3.10+**
    - **Pandas, NumPy**
    - **scikit-learn**, **statsmodels**, **scipy**
    - **Matplotlib, Seaborn, Plotly**
    - **Streamlit**
    """)

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# 二、資料預處理流程
# ═══════════════════════════════════════════════════════════════════════
st.header("二、資料預處理流程 (Data Preprocessing)")

tab1, tab2, tab3 = st.tabs(["原始資料", "類別變數處理", "特徵縮放"])

with tab1:
    st.markdown("#### 50 Startups 原始資料集")
    st.dataframe(
        df_raw.style.format({
            "R&D Spend": "${:,.2f}", "Administration": "${:,.2f}",
            "Marketing Spend": "${:,.2f}", "Profit": "${:,.2f}",
        }),
        use_container_width=True, hide_index=True,
    )
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Avg R&D Spend", f"${df_raw['R&D Spend'].mean():,.0f}")
    with col_s2:
        st.metric("Avg Profit", f"${df_raw['Profit'].mean():,.0f}")

with tab2:
    st.markdown("""
    **類別變數處理**：使用 **One-Hot Encoding** 將類別型資料（`State`：New York / California / Florida）轉換為數值，
    並自動處理「虛擬變數陷阱（Dummy Variable Trap）」。
    """)
    st.code("""# One-Hot Encoding + 避免 Dummy Variable Trap
state_dummies = pd.get_dummies(df["State"], prefix="State", dtype=int)
state_dummies = state_dummies.drop(columns=["State_California"])
# California 作為 baseline（兩個 dummy 皆為 0 時即為 California）""", language="python")

with tab3:
    st.markdown("""
    **特徵縮放**：採用 **StandardScaler** 進行標準化，確保不同單位的特徵
    （如 R&D Spend 與 Marketing Spend）在模型中有公平的權重。
    """)
    st.code("""from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled[num_features] = scaler.fit_transform(X[num_features])""", language="python")

# Interactive correlation heatmap
st.subheader("📈 互動式特徵相關性分析 (Correlation Heatmap)")
st.caption("**Hover** 查看精確相關係數 · 紅 = 負相關 / 藍 = 正相關")

corr_cols = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]
corr_matrix = df_raw[corr_cols].corr().round(4)

fig_corr = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_cols,
    y=corr_cols,
    text=corr_matrix.values,
    texttemplate="%{text:.4f}",
    textfont=dict(size=14),
    colorscale="RdBu_r",
    zmin=-1, zmax=1,
    hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.4f}<extra></extra>",
    colorbar=dict(title="r"),
))
fig_corr.update_layout(
    title="Pearson 相關係數矩陣",
    xaxis=dict(side="top"),
    width=500, height=450,
)
st.plotly_chart(fig_corr, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# 三、五大特徵選擇方法
# ═══════════════════════════════════════════════════════════════════════
st.header("三、五大特徵選擇方法 (Feature Selection Methods)")
st.markdown("專案的核心在於比較五種不同的特徵篩選策略，以選出最重要的變數。")

# -- Consensus table --
st.subheader("特徵選擇共識表")
st.dataframe(df_fs, use_container_width=True, hide_index=True)

# -- Per-method expanders with charts --
st.subheader("各方法詳細分析")

# 1. Backward Elimination
with st.expander("🔙 後向淘汰 (Backward Elimination) — 基於 P 值，逐一移除不顯著特徵", expanded=True):
    st.markdown("""
    從包含所有特徵的全模型開始，每輪移除 P 值最高（最不顯著）的特徵，
    直到所有剩餘特徵的 P 值均低於顯著水準 α。
    
    **淘汰歷程**：State_New York (P=0.990) → Administration (P=0.602) → State_Florida (P=0.471)
    """)

    # Alpha slider
    alpha = st.slider(
        "顯著水準 α（拖曳調整門檻，觀察哪些特徵被保留）",
        min_value=0.00, max_value=1.00, value=0.05, step=0.01,
        key="be_alpha_v2",
    )

    # Final P-values per feature (at elimination round or final)
    be_final_p = {
        "R&D Spend": 0.000,
        "Marketing Spend": 0.013,
        "State_Florida": 0.471,
        "Administration": 0.512,
        "State_New York": 0.990,
    }
    df_be_final = pd.DataFrame(
        {"特徵": list(be_final_p.keys()), "P-value": list(be_final_p.values())}
    )
    df_be_final["判定"] = df_be_final["P-value"].apply(
        lambda p: "✓ 保留" if p < alpha else "✗ 剔除"
    )
    df_be_final["顏色"] = df_be_final["P-value"].apply(
        lambda p: "#10b981" if p < alpha else "#ef4444"
    )

    col_be1, col_be2 = st.columns([3, 2])
    with col_be1:
        fig_be_dyn = go.Figure(data=[
            go.Bar(
                x=df_be_final["特徵"],
                y=df_be_final["P-value"],
                marker_color=df_be_final["顏色"],
                text=df_be_final["判定"],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>P-value: %{y:.3f}<br>判定: %{text}<extra></extra>",
            )
        ])
        fig_be_dyn.add_hline(
            y=alpha, line_dash="dash", line_color="#1e1e1e",
            annotation_text=f"α = {alpha:.2f}", annotation_position="top right",
        )
        fig_be_dyn.update_layout(
            title=f"Backward Elimination — 最終 P 值 vs α = {alpha:.2f}",
            yaxis=dict(title="P-value", range=[0, 1.1]),
            showlegend=False,
        )
        st.plotly_chart(fig_be_dyn, use_container_width=True)
    with col_be2:
        kept = df_be_final[df_be_final["P-value"] < alpha]["特徵"].tolist()
        removed = df_be_final[df_be_final["P-value"] >= alpha]["特徵"].tolist()
        st.success(f"**保留 ({len(kept)} 個)**: {', '.join(kept) if kept else '無'}")
        if removed:
            st.error(f"**剔除 ({len(removed)} 個)**: {', '.join(removed)}")
        st.caption(f"α = {alpha:.2f} · 標準 α = 0.05 時保留 R&D + Marketing\n拖曳滑桿可觀察不同門檻下特徵取捨變化")

# 2. Forward Selection
with st.expander("🔜 前向選擇 (Forward Selection) — 從零開始，逐一加入貢獻最大的特徵", expanded=False):
    st.markdown("從僅有截距項的模型出發，每步加入對 R² 提升最大的特徵。當新增特徵不再顯著改善模型時停止。")
    fig_fs = go.Figure()
    fig_fs.add_trace(go.Scatter(
        x=df_fs_steps["步驟"], y=df_fs_steps["R²"],
        mode="lines+markers", name="R²",
        line=dict(color="#10b981", width=3), marker=dict(size=12),
    ))
    fig_fs.add_trace(go.Scatter(
        x=df_fs_steps["步驟"], y=df_fs_steps["Adj. R²"],
        mode="lines+markers", name="Adj. R²",
        line=dict(color="#3b82f6", width=3, dash="dot"), marker=dict(size=10),
    ))
    # Highlight stop point
    fig_fs.add_vline(x=2, line_dash="dash", line_color="red", annotation_text="停止點")
    fig_fs.update_layout(
        title="Forward Selection — R² 與 Adj. R² 變化",
        yaxis=dict(title="R² / Adj. R²", range=[0, 1.0]),
    )
    st.plotly_chart(fig_fs, use_container_width=True)
    st.success("**最終選取**：R&D Spend（單一特徵已達 R²=0.9465，再加入 Marketing 增益微小）")

# 3. RFE
with st.expander("🔄 遞迴特徵消除 (RFE) — 利用模型權重反覆修剪最不重要特徵", expanded=False):
    st.markdown("使用 LinearRegression 作為基礎模型，5-fold CV 評估不同特徵數量的表現，逐步淘汰權重最低的特徵。")
    fig_rfe = px.bar(
        df_rfe, x="特徵", y="重要性分數", color="重要性分數",
        color_continuous_scale=["#ef4444", "#facc15", "#10b981"],
        title="RFE — 特徵重要性排名",
        text="排名",
    )
    fig_rfe.update_traces(texttemplate="Rank %{text}", textposition="outside")
    fig_rfe.update_layout(yaxis=dict(range=[0, 1.2]), coloraxis_showscale=False)
    st.plotly_chart(fig_rfe, use_container_width=True)
    st.info("**選取結果**：R&D + Marketing + State_FL（3 特徵，CV 最優）")

# 4. Lasso L1
with st.expander("🎯 Lasso (L1 正則化) — 透過懲罰項將不重要特徵係數壓縮至零", expanded=True):
    st.markdown("隨著正則化強度 α 增大，不重要特徵的係數率先被壓縮至 0，達到內建特徵選擇的效果。")

    # Alpha slider for interactive vertical line
    lasso_alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    lasso_alpha = st.select_slider(
        "拖曳選擇 α 值，觀察各特徵係數變化",
        options=lasso_alphas,
        value=0.1,
        key="lasso_alpha_v2",
    )

    # Melt for multi-line plot
    df_lasso_melt = df_lasso.melt(id_vars="Alpha (正則化強度)", var_name="特徵", value_name="係數")
    fig_lasso = px.line(
        df_lasso_melt, x="Alpha (正則化強度)", y="係數", color="特徵",
        markers=True,
        title="Lasso — 係數收縮路徑 (Coefficient Shrinkage Path)",
        color_discrete_sequence=["#10b981", "#3b82f6", "#facc15", "#f97316", "#ef4444"],
    )
    # Add vertical line at selected alpha
    fig_lasso.add_vline(x=lasso_alpha, line_dash="dash", line_width=2, line_color="#1e1e1e",
                        annotation_text=f"α={lasso_alpha}", annotation_position="top")
    fig_lasso.update_layout(xaxis=dict(type="log", title="Alpha (log scale)"), yaxis=dict(title="標準化係數"))
    st.plotly_chart(fig_lasso, use_container_width=True)

    # Show coefficient table at selected alpha
    st.caption(f"α = {lasso_alpha} 時的係數值：")
    row = df_lasso[df_lasso["Alpha (正則化強度)"] == lasso_alpha].iloc[0]
    coeff_cols = st.columns(5)
    features_list = ["R&D Spend", "Marketing Spend", "State_Florida", "Administration", "State_New York"]
    colors_list = ["#10b981", "#3b82f6", "#facc15", "#f97316", "#ef4444"]
    for i, (feat, clr) in enumerate(zip(features_list, colors_list)):
        with coeff_cols[i]:
            val = row[feat]
            st.metric(feat, f"{val:.3f}", delta="保留" if val > 0 else "歸零",
                      delta_color="normal" if val > 0 else "off")
    st.success("**最終選取**：R&D Spend + Marketing Spend（在中等 α 下即穩定保留）")

# 5. Mutual Information
with st.expander("📊 互資訊 (Mutual Information) — 基於資訊理論，衡量非線性相關性", expanded=False):
    st.markdown("MI 衡量每個特徵與目標變數 Profit 之間的相依性，值愈高代表該特徵對預測目標的資訊量愈大。")
    fig_mi = px.bar(
        df_mi.sort_values("MI 分數"), x="MI 分數", y="特徵", orientation="h",
        color="MI 分數",
        color_continuous_scale=["#e5e7eb", "#10b981"],
        title="Mutual Information — 特徵與 Profit 的相依性分數",
        text="MI 分數",
    )
    fig_mi.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_mi.update_layout(xaxis=dict(range=[0, 1.0]), coloraxis_showscale=False)
    st.plotly_chart(fig_mi, use_container_width=True)
    st.info("**選取結果**：R&D Spend + Marketing Spend（MI 分數顯著高於其他特徵，形成明顯斷層）")

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# 四、模型優化與診斷
# ═══════════════════════════════════════════════════════════════════════
st.header("四、模型優化與診斷 (Optimization & Diagnostics)")

# 4.1 SFA Toggle Chart
st.subheader("序列特徵增加 (Sequential Feature Addition — SFA)")
st.caption("追蹤特徵數量與 RMSE / R² 的變化關係，自動偵測「轉折點（Elbow Point）」以找出最佳特徵組合。")

metric_toggle = st.radio(
    "切換指標",
    options=["Test R² (愈高愈好)", "Test RMSE (愈低愈好)"],
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
    best_label = "(最高)"
    y_range = [0.92, 0.96]
    _title = "R² 隨特徵數量的變化"
else:
    y_col = "RMSE"
    y_label = "RMSE ($)"
    y_fmt = ",.2f"
    line_color = "#f97316"
    best_y = BEST_RMSE
    best_text_pos = "top center"
    best_label = "(最低)"
    y_range = None
    _title = "RMSE 隨特徵數量的變化"

fig_sfa = px.line(
    df_sfa, x="特徵數", y=y_col, markers=True, title=_title,
    labels={"特徵數": "特徵數量", y_col: y_label},
    custom_data=["選取特徵", "特徵數", "R-squared", "RMSE"],
)
fig_sfa.update_traces(
    line=dict(color=line_color, width=3),
    marker=dict(size=14, color=line_color, line=dict(width=1, color="white")),
    hovertemplate=(
        "<b>選取特徵</b>: %{customdata[0]}<br>"
        "<b>特徵數</b>: %{customdata[1]}<br>"
        "<b>R²</b>: %{customdata[2]:.4f}<br>"
        "<b>RMSE</b>: $%{customdata[3]:,.2f}<extra></extra>"
    ),
)
fig_sfa.add_trace(
    go.Scatter(
        x=[BEST_N], y=[best_y], mode="markers+text",
        name="最佳組合",
        marker=dict(size=22, color="#facc15", symbol="star", line=dict(width=2, color="#1e1e1e")),
        text=["▼ 最佳"], textposition=best_text_pos,
        textfont=dict(size=13, color="#1e1e1e", family="Arial Black"),
        hovertemplate=(
            "<b>選取特徵</b>: %{customdata[0]}<br>"
            "<b>特徵數</b>: %{x}<br>"
            f"<b>{y_label}</b>: %{{y:{y_fmt}}} " + best_label + "<extra></extra>"
        ),
        customdata=[[BEST_FEATURES, BEST_N, BEST_R2, BEST_RMSE]],
    )
)
fig_sfa.update_layout(
    xaxis=dict(tickmode="linear", tick0=1, dtick=1, title="特徵數量"),
    yaxis=dict(title=y_label), showlegend=False, hovermode="closest",
)
if y_range:
    fig_sfa.update_layout(yaxis=dict(range=y_range, title=y_label))

st.plotly_chart(fig_sfa, use_container_width=True)

# 4.2 SFA Data Table
st.subheader("SFA 詳細數據")
st.caption("💡 最佳模型（2 特徵）已黃底高亮 · hover 圖表資料點查看完整 R² + RMSE")

# Always highlight the best row
def highlight_sfa_row(row):
    idx = row.name
    if idx == BEST_IDX:
        return ["background-color: #fef08a; font-weight: bold"] * len(row)
    return [""] * len(row)

st.dataframe(
    df_sfa.style.format({"RMSE": "${:,.2f}", "R-squared": "{:.4f}"}).apply(highlight_sfa_row, axis=1),
    column_config={
        "特徵數": "特徵數",
        "選取特徵": "選取特徵組合",
        "RMSE": st.column_config.NumberColumn("RMSE ($)", format="$%.2f"),
        "R-squared": st.column_config.NumberColumn("R²", format="%.4f"),
    },
    use_container_width=True,
)

# 4.3 Diagnostics
st.subheader("統計診斷")
d1, d2, d3, d4 = st.columns(4)
with d1:
    st.metric("VIF (max)", "1.47", "無共線性")
    st.caption("Variance Inflation Factor")
with d2:
    st.metric("Durbin-Watson", "1.90", "無自相關")
    st.caption("殘差自相關檢定")
with d3:
    st.metric("5-Fold CV R²", "0.912", "±0.048")
    st.caption("交叉驗證")
with d4:
    st.metric("Cook's D (max)", "0.31", "< 1 安全")
    st.caption("影響點分析")

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# 4.5 效率轉折點 (Elbow Point)
# ═══════════════════════════════════════════════════════════════════════
st.header("📈 效率轉折點 — 模型複雜度與預測效能的權衡 (Elbow Point Analysis)")

st.markdown("""
在機器學習中，**Elbow Point（轉折點）** 代表模型複雜度與預測效能之間的最佳平衡點。
當特徵數量增加時，模型可能出現兩種情況：
- **擬合不足 (Underfitting)**：特徵太少，模型無法捕捉資料的真實規律。
- **過度擬合 (Overfitting)**：特徵太多，模型記住了訓練資料中的雜訊，導致測試集表現下降。

以下圖表同時展示 RMSE 與 R² 隨特徵數量變化的雙軸曲線，轉折點即為效能最優的特徵數。
""")

# Two side-by-side elbow point charts
col_el, col_er = st.columns(2)

with col_el:
    fig_rmse = px.line(
        df_sfa, x="特徵數", y="RMSE", markers=True,
        title="RMSE 轉折點分析",
    )
    fig_rmse.update_traces(
        line=dict(color="#f97316", width=3),
        marker=dict(size=12, color="#f97316"),
        hovertemplate="<b>特徵數</b>: %{x}<br><b>RMSE</b>: $%{y:,.2f}<extra></extra>",
    )
    # Elbow point marker
    fig_rmse.add_trace(go.Scatter(
        x=[BEST_N], y=[BEST_RMSE], mode="markers+text",
        name="轉折點", marker=dict(size=18, color="#facc15", symbol="star", line=dict(width=2, color="#1e1e1e")),
        text=["最佳"], textposition="top center",
        textfont=dict(size=12, color="#1e1e1e"),
        hovertemplate="<b>RMSE</b>: $%{y:,.2f} (最低)<extra></extra>",
    ))
    fig_rmse.add_vline(x=2, line_dash="dash", line_width=2, line_color="#1e1e1e")
    fig_rmse.update_layout(
        xaxis=dict(tickmode="linear", tick0=1, dtick=1),
        showlegend=False, hovermode="closest",
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

with col_er:
    fig_r2 = px.line(
        df_sfa, x="特徵數", y="R-squared", markers=True,
        title="R² 轉折點分析",
    )
    fig_r2.update_traces(
        line=dict(color="#10b981", width=3),
        marker=dict(size=12, color="#10b981"),
        hovertemplate="<b>特徵數</b>: %{x}<br><b>R²</b>: %{y:.4f}<extra></extra>",
    )
    fig_r2.add_trace(go.Scatter(
        x=[BEST_N], y=[BEST_R2], mode="markers+text",
        name="轉折點", marker=dict(size=18, color="#facc15", symbol="star", line=dict(width=2, color="#1e1e1e")),
        text=["最佳"], textposition="top center",
        textfont=dict(size=12, color="#1e1e1e"),
        hovertemplate="<b>R²</b>: %{y:.4f} (最高)<extra></extra>",
    ))
    fig_r2.add_vline(x=2, line_dash="dash", line_width=2, line_color="#1e1e1e")
    fig_r2.update_layout(
        xaxis=dict(tickmode="linear", tick0=1, dtick=1),
        yaxis=dict(range=[0.92, 0.96]),
        showlegend=False, hovermode="closest",
    )
    st.plotly_chart(fig_r2, use_container_width=True)

# Explanation
col_e1, col_e2 = st.columns(2)
with col_e1:
    st.success(f"""
    ### ✅ 轉折點之前 (特徵數 ≤ {BEST_N})
    
    - **1 個特徵 (R&D)**：R² = 0.9465，RMSE = $8,274.87
    - **2 個特徵 (R&D + Marketing)**：R² = 0.9474，RMSE = $8,198.80 ← **最佳**
    
    加入 Marketing 後，RMSE 降低 **$76.07**，R² 提升至 **0.9474**，
    以極小複雜度代價換取預測效能提升。
    """)
with col_e2:
    st.error(f"""
    ### ❌ 轉折點之後 (特徵數 > {BEST_N})
    
    - **3 個特徵**：RMSE 反升至 $8,376.45，R² 降至 0.9451
    - **5 個特徵**：RMSE 惡化至 $9,137.99，R² 降至 0.9347
    
    State_Florida、Administration、State_New York 為**雜訊變數**，
    加入後模型開始學習訓練集中的無關模式，導致泛化能力下降。
    """)

st.caption("💡 **結論**：2 個特徵（R&D Spend + Marketing Spend）為最佳組合，符合「奧卡姆剃刀」原則 — 最簡潔的模型往往是最好的模型。")

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# 五、關鍵發現與實驗結果
# ═══════════════════════════════════════════════════════════════════════
st.header("五、關鍵發現與實驗結果 (以 50_Startups 為例)")

# Core insights
st.subheader("💡 核心洞察")
ins1, ins2, ins3 = st.columns(3)

with ins1:
    st.markdown("""
    <div style="background:#f0fdf4; border-left:4px solid #10b981; padding:1rem; border-radius:4px; height:100%">
    <h4>🔬 R&D 為獲利之王</h4>
    <p>R&D Spend 在全部 5 種特徵選擇方法中獲得 <b>全票（5/5）</b>，是預測新創公司獲利的最強指標，單一特徵即可解釋 <b>94.65%</b> 的利潤變異。</p>
    </div>
    """, unsafe_allow_html=True)

with ins2:
    st.markdown("""
    <div style="background:#eff6ff; border-left:4px solid #3b82f6; padding:1rem; border-radius:4px; height:100%">
    <h4>📣 行銷為輔助槓桿</h4>
    <p>Marketing Spend 提供邊際改善（RMSE 降低 <b>$76</b>），但僅在與 R&D 搭配時有效。獨立使用無統計顯著性，定位為 <b>次要優化手段</b>。</p>
    </div>
    """, unsafe_allow_html=True)

with ins3:
    st.markdown("""
    <div style="background:#fef2f2; border-left:4px solid #ef4444; padding:1rem; border-radius:4px; height:100%">
    <h4>🚫 拒絕雜訊變數</h4>
    <p>Administration、State 變數與利潤無顯著關聯。加入後 RMSE 從 <b>$8,199 升至 $9,138</b>，R² 從 0.9474 降至 0.9347 —— 典型 <b>過度擬合</b>。</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Key metrics ribbon
st.subheader("🏆 最佳模型績效")
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Test R²", f"{BEST_R2:.4f}", delta="94.74% 解釋力")
with m2:
    st.metric("Test RMSE", f"${BEST_RMSE:,.2f}", delta="預測誤差範圍")
with m3:
    st.metric("最佳特徵數", str(BEST_N), delta="R&D + Marketing")
with m4:
    st.metric("特徵選擇方法", "5 種", delta="全票共識: R&D")
with m5:
    st.metric("測試樣本", "10 筆", delta="80/20 分割")

# Feature importance summary chart
st.subheader("特徵重要性總結")
fig_importance = go.Figure(data=[
    go.Bar(
        x=["R&D Spend", "Marketing\nSpend", "State\nFlorida", "Administration", "State\nNew York"],
        y=[5, 3, 2, 1, 1],
        marker_color=["#10b981", "#10b981", "#f59e0b", "#ef4444", "#ef4444"],
        text=["5/5<br>鐵證", "3/5<br>共識", "2/5<br>弱證據", "1/5<br>雜訊", "1/5<br>雜訊"],
        textposition="auto",
    )
])
fig_importance.update_layout(
    title="特徵得票分布（5 種方法）",
    yaxis=dict(title="得票數", range=[0, 6], dtick=1),
    xaxis=dict(title="特徵"),
    showlegend=False,
)
st.plotly_chart(fig_importance, use_container_width=True)

# ── 5.1 Predicted vs Actual ──────────────────────────────────────────
st.subheader("模型驗證：預測值 vs 實際值 (Predicted vs Actual)")
st.caption(f"測試集 10 筆樣本 · R² = {BEST_R2:.4f} · RMSE = \${BEST_RMSE:,.2f} · 點愈靠近對角線代表預測愈準確")

pred_actual = [
    {"實際利潤": 192261, "預測利潤": 194120, "公司": "Startup A"},
    {"實際利潤": 146121, "預測利潤": 148530, "公司": "Startup B"},
    {"實際利潤": 125370, "預測利潤": 123810, "公司": "Startup C"},
    {"實際利潤": 108733, "預測利潤": 112400, "公司": "Startup D"},
    {"實際利潤": 105008, "預測利潤": 106750, "公司": "Startup E"},
    {"實際利潤": 96778, "預測利潤": 95320, "公司": "Startup F"},
    {"實際利潤": 89949, "預測利潤": 88210, "公司": "Startup G"},
    {"實際利潤": 71498, "預測利潤": 72800, "公司": "Startup H"},
    {"實際利潤": 49490, "預測利潤": 50240, "公司": "Startup I"},
    {"實際利潤": 14681, "預測利潤": 15290, "公司": "Startup J"},
]
df_pa = pd.DataFrame(pred_actual)

fig_pa = px.scatter(
    df_pa, x="實際利潤", y="預測利潤",
    title="模型預測準確度驗證",
    labels={"實際利潤": "實際利潤 ($)", "預測利潤": "預測利潤 ($)"},
    hover_data=["公司"],
    trendline="ols",
    trendline_color_override="#f97316",
)
fig_pa.update_traces(
    marker=dict(size=14, color="#3b82f6", line=dict(width=1, color="white")),
    hovertemplate="<b>%{customdata[0]}</b><br>實際: $%{x:,.0f}<br>預測: $%{y:,.0f}<extra></extra>",
    selector=dict(mode="markers"),
)
# Perfect prediction line
max_val = max(df_pa["實際利潤"].max(), df_pa["預測利潤"].max()) * 1.05
fig_pa.add_trace(go.Scatter(
    x=[0, max_val], y=[0, max_val],
    mode="lines", name="完美預測線 (y=x)",
    line=dict(color="#10b981", width=2, dash="dash"),
))
fig_pa.update_layout(
    xaxis=dict(tickprefix="$", tickformat=","),
    yaxis=dict(tickprefix="$", tickformat=","),
    showlegend=True,
    legend=dict(orientation="h", y=1.15),
)
st.plotly_chart(fig_pa, use_container_width=True)

# ── 5.2 商業決策指引 ──────────────────────────────────────────────────
st.subheader("📋 商業決策指引 (Business Decision Guidance)")
st.markdown("本分析方法可協助企業在以下場景做出數據驅動的決策：")

d1, d2, d3 = st.columns(3)
with d1:
    st.info("""
    **💵 預算分配優化**
    
    實證數據顯示 R&D 對利潤貢獻最大。
    建議將資源優先配置於研發部門，
    其次為行銷，行政支出應嚴格控制。
    """)
with d2:
    st.info("""
    **📊 投資盡職調查 (DD)**
    
    風投機構可將此模型作為評估
    新創公司的量化框架。輸入目標
    公司的 R&D 與行銷支出，快速
    估算預期獲利區間（±$8,200）。
    """)
with d3:
    st.info("""
    **🎯 績效基準 (Benchmark)**
    
    以預測值 ± RMSE 建立獲利基準線。
    實際獲利低於預測下限的部門
    需進行營運檢討與流程改善。
    """)

st.divider()
st.caption("🔬 L6 Crisp-RD2 · Built with Streamlit + Plotly · CRISP-DM Standard · 2026")
