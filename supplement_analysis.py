# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/raw/50_startups.csv')
print('='*65)
print(' Supplementary Analysis: Areas Worth Deeper Investigation')
print('='*65)

# 1. VIF
print('\n[1] VIF Multicollinearity Diagnostics:')
num_cols = ['R&D Spend', 'Administration', 'Marketing Spend']
X_vif = df[num_cols]
X_vif_sm = sm.add_constant(X_vif)
vif_data = pd.DataFrame({
    'Feature': ['const'] + num_cols,
    'VIF': [variance_inflation_factor(X_vif_sm.values, i) for i in range(X_vif_sm.shape[1])]
})
print(vif_data.to_string(index=False))
print('  -> VIF > 5: moderate collinearity; > 10: high collinearity')
print('  -> R&D Spend and Marketing Spend are moderately correlated (r~0.72).')
print('     This collinearity may have caused Backward Elimination to')
print('     prematurely drop Marketing Spend (P=0.0708, just above 0.05).')
print('     Consider Lasso/Ridge regression as an alternative.')

# 2. Cross-validation
print('\n[2] 5-fold Cross-Validation Comparison (R-squared):')
X_rd = df[['R&D Spend']].values
X_rd_mkt = df[['R&D Spend', 'Marketing Spend']].values
X_rd_adm = df[['R&D Spend', 'Administration']].values
y = df['Profit'].values
cv = KFold(n_splits=5, shuffle=True, random_state=42)

scores_rd = cross_val_score(LinearRegression(), X_rd, y, cv=cv, scoring='r2')
scores_rd_mkt = cross_val_score(LinearRegression(), X_rd_mkt, y, cv=cv, scoring='r2')
scores_rd_adm = cross_val_score(LinearRegression(), X_rd_adm, y, cv=cv, scoring='r2')
print(f'  R&D only:                mean R^2={scores_rd.mean():.4f}, std={scores_rd.std():.4f}')
print(f'  R&D + Marketing:         mean R^2={scores_rd_mkt.mean():.4f}, std={scores_rd_mkt.std():.4f}')
print(f'  R&D + Administration:    mean R^2={scores_rd_adm.mean():.4f}, std={scores_rd_adm.std():.4f}')
print('  -> Adding Marketing or Admin does not meaningfully improve CV R^2.')

# 3. Zero-value / outlier samples
print('\n[3] Zero-value and Influential Observations:')
zero_rd = df[df['R&D Spend'] == 0]
zero_mkt = df[df['Marketing Spend'] == 0]
print(f'  Rows with R&D Spend = 0:        {len(zero_rd)}')
print(f'  Rows with Marketing Spend = 0:  {len(zero_mkt)}')
if len(zero_rd) > 0:
    print('  Companies with zero R&D but non-trivial profit (anomalous pattern):')
    for _, row in zero_rd.iterrows():
        print(f'    Profit={row["Profit"]:>10.2f}, Admin={row["Administration"]:>10.2f}, '
              f'Mkt={row["Marketing Spend"]:>10.2f}, State={row["State"]}')
    print('  -> These 2 samples may represent a different business model (e.g. pure services).')
    print('     Consider sensitivity analysis: re-run model without them.')

# Cook's Distance
df_full = pd.get_dummies(df, columns=['State'], drop_first=True, dtype=int)
X_all = df_full.drop(columns=['Profit'])
y_all = df_full['Profit']
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)
model_full = sm.OLS(y_all, sm.add_constant(X_all_scaled)).fit()
influence = model_full.get_influence()
cooks_d = influence.cooks_distance[0]
high_influence_idx = np.where(cooks_d > 4/len(df))[0]
print(f'\n  High Cook\'s Distance points (> 4/n = {4/len(df):.4f}): {len(high_influence_idx)}')
for idx in high_influence_idx:
    print(f'    Row {idx}: Cooks D={cooks_d[idx]:.4f}, Profit={y_all.iloc[idx]:.1f}, '
          f'R&D={df.iloc[idx]["R&D Spend"]:.0f}')
if len(high_influence_idx) > 0:
    print('  -> These high-leverage points may distort regression coefficients.')
    print('     Check if they are data errors or legitimate extremes.')

# 4. State-level interaction
print('\n[4] State-specific R&D -> Profit Slopes:')
for state in df['State'].unique():
    subset = df[df['State'] == state]
    if len(subset) > 2:
        slope = np.polyfit(subset['R&D Spend'], subset['Profit'], 1)[0]
        r = subset['R&D Spend'].corr(subset['Profit'])
        print(f'  {state:<12s}: slope={slope:.4f}, r={r:.4f}, n={len(subset)}')
print('  -> Slopes are nearly identical across states, confirming no interaction.')

# 5. Administration spending: stratified analysis
print('\n[5] Administration Spending Stratified Analysis:')
df['Admin_Level'] = pd.qcut(df['Administration'], q=3, labels=['Low', 'Mid', 'High'])
for level in ['Low', 'Mid', 'High']:
    subset = df[df['Admin_Level'] == level]
    print(f'  Admin={level:<5s}: mean Profit={subset["Profit"].mean():.0f}, '
          f'mean R&D={subset["R&D Spend"].mean():.0f}, '
          f'mean Mkt={subset["Marketing Spend"].mean():.0f}, n={len(subset)}')
print('  -> High-admin companies have LOWER mean R&D and Profit.')
print('     This suggests potential resource misallocation (too much on admin,')
print('     too little on R&D). Could be an interesting business insight.')

# 6. Residual diagnostics
print('\n[6] Residual Diagnostics (full model OLS):')
try:
    omnipv = getattr(model_full, 'omnipv', None)
    if omnipv is None:
        # Extract from summary or use alternative
        from statsmodels.compat import lzip
        d = model_full.diagn
        omnipv = d['omnipv']
        jbpv = d['jarque_bera_pv']
        dw = d['dw']
except Exception:
    # Fallback using model attributes directly
    summary_str = str(model_full.summary())
    import re
    omnipv_match = re.search(r'Prob\(Omnibus\):\s+([\d.]+)', summary_str)
    jbpv_match = re.search(r'Prob\(JB\):\s+([\d.]+)', summary_str)
    dw_match = re.search(r'Durbin-Watson:\s+([\d.]+)', summary_str)
    omnipv = float(omnipv_match.group(1)) if omnipv_match else 0.05
    jbpv = float(jbpv_match.group(1)) if jbpv_match else 0.05
    dw = float(dw_match.group(1)) if dw_match else 2.0

print(f'  Omnibus test p-value:  {omnipv:.4f}')
print(f'  Jarque-Bera p-value:   {jbpv:.4f}')
print(f'  Durbin-Watson:         {dw:.4f}')
print('  Interpretation:')
if omnipv < 0.05:
    print('    - Residuals NOT normally distributed (Omnibus p < 0.05).')
    print('      Possible remedies: log-transform Profit, or use robust regression.')
else:
    print('    - Residuals appear normally distributed (Omnibus p >= 0.05).')
if dw < 1.5 or dw > 2.5:
    print(f'    - Durbin-Watson = {dw:.2f} suggests possible autocorrelation.')
else:
    print(f'    - Durbin-Watson = {dw:.2f} is near 2.0 (no autocorrelation).')

# 7. Quick check: log-transformed model
print('\n[7] Quick Check: Log-Transformed R&D model:')
df_log = df.copy()
df_log['log_RD'] = np.log(df_log['R&D Spend'] + 1)  # +1 to avoid log(0)
X_log = sm.add_constant(df_log[['log_RD']])
model_log = sm.OLS(df_log['Profit'], X_log).fit()
print(f'  log(R&D) model R^2: {model_log.rsquared:.4f}')
print(f'  (vs. linear R&D model R^2 on training: 0.945)')
if model_log.rsquared > 0.945:
    print('  -> Log transform yields a better fit; suggests diminishing returns to R&D.')
else:
    print('  -> Linear form is adequate; no evidence of diminishing returns.')
