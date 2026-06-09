# Changelog

All notable changes to the L6 Crisp-RD2 ML Regression Pipeline project.

---

## [1.7] — 2026-06-09

### Added
- White paper Chapter 11: Changelog
- White paper Section 9 updated with full dashboard block details

## [1.6] — 2026-06-09

### Added
- `ml-regression-pipeline` OpenCode Skill updated to 16-step workflow
- Skill includes: design.md, 5 methods code, diagnostics checklist, dashboard deployment, white paper generation, README format

## [1.5] — 2026-06-09

### Added
- GitHub Pages dashboard: Method Comparison Results HTML table (5 methods, full metrics)
- GitHub Pages dashboard: Sequential Feature Addition Results HTML table (1→5 features)
- GitHub Pages dashboard: Feature Votes cards with progress bars (5 cards)

## [1.4] — 2026-06-09

### Added
- `design.md` — Architecture & pipeline specification
- `supplement_analysis.py` — VIF, CV, Cook's D diagnostics script
- `feature_selection.py` — 5-method feature selection comparison
- `refined_models.py` — Outlier removal + Box-Cox + Huber regression
- `outcome_visualization.py` — Sequential feature addition charts
- `method_comparison_charts.py` — 5-method comparison dashboard charts

## [1.3] — 2026-06-09

### Added
- `README.md` with badges (License, Python, scikit-learn, GitHub Pages) and Live Demo links
- `.gitignore` — excludes .env, Python cache, venv, IDE files

## [1.2] — 2026-06-09

### Added
- `WHITEPAPER.md` — 10-chapter technical white paper covering:
  - Project overview, system architecture, data specs
  - 5 feature selection methodology, model implementation
  - Experimental results, model diagnostics
  - Deployment architecture, appendix

## [1.1] — 2026-06-09

### Added
- `index.html` — GitHub Pages interactive dashboard (Tailwind CSS, dark theme)
- `.github/workflows/deploy.yml` — GitHub Actions CI/CD for Pages deployment
- 11 PNG figures in `outputs/figures/`
- 6 reports (txt/csv/tsv) in `outputs/reports/`

## [1.0] — 2026-06-09

### Initial Release
- `main_analysis.py` — Full ML pipeline (correlation → preprocessing → backward elimination → evaluation)
- Data preprocessing with One-Hot Encoding and dummy variable trap avoidance
- Backward Elimination via statsmodels OLS (P-value threshold)
- Model evaluation (R², Adj. R², MAE, RMSE, residual diagnostics)
- `data/raw/50_startups.csv` — Source dataset (50 rows, 5 columns)
- `data/processed/` — Train/test split CSVs
- `requirements.txt` — Python dependencies
- `ml-regression-pipeline` OpenCode Skill (initial version)
- `skill-creator` OpenCode Skill (from Anthropic)
