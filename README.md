# 🌽 Climate Variability & Maize Yield in Uasin Gishu, Kenya
## Fourth Year — Applied Statistics & Computing — Final Project
### Data Source: Ministry of Agriculture & Livestock Development, Kenya

---

## 📖 Project Synopsis

This project uses **real station data from Eldoret, Uasin Gishu County** to investigate
the statistical relationship between climate variability (rainfall and temperature) and
maize yield over 35 years. The study applies a rigorous applied statistics pipeline:
descriptive analysis → trend testing → hypothesis testing → regression modelling →
an interactive Streamlit dashboard.

**Key honest finding:** In Uasin Gishu, climate variables (rainfall, temperature)
explain very little of the annual variation in maize yield — pointing to agronomic
management as the dominant driver. This is itself an important, publishable result.

---

## 🗂️ Project Structure

```
real_project/
├── app.py                        ← Streamlit dashboard (8 pages)
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
│
├── data/
│   ├── rainfall_monthly.csv      ← 432 rows (1990–2025, 12 months × 36 years)
│   ├── rainfall_annual.csv       ← 36 rows — annual totals & stats
│   ├── temperature_monthly.csv   ← 372 rows (1995–2025)
│   ├── temperature_annual.csv    ← 31 rows — annual means
│   ├── maize_yield.csv           ← 12 rows (2012–2023, Uasin Gishu)
│   └── panel_annual.csv          ← 12 rows — merged climate + yield panel
│
└── figures/                      ← 6 publication-quality figures
    ├── fig1_eda.png
    ├── fig2_hypothesis.png
    ├── fig3_regression.png
    ├── fig4_climate_trends.png
    ├── fig5_correlation.png
    └── fig6_dashboard.png
```

---

## 📦 Data Sources

| Dataset | Period | N | Source |
|---|---|---|---|
| Monthly Rainfall | 1990–2025 | 432 records | RAINFALL_DATA_FROM_1990-2025.docx |
| Monthly Temperature | 1995–2025 | 372 records | TEMPERATURE_DATA_FROM_1995.docx |
| Maize Yield | 2012–2023 | 12 records | crops_data.csv + Project_Data.xlsx |

**Station:** Eldoret Meteorological Station, Uasin Gishu County  
**Authority:** Ministry of Agriculture and Livestock Development, Kenya

---

## 🔬 Statistical Methods

| Section | Method | Purpose |
|---|---|---|
| Descriptive Stats | Mean, SD, CV, skewness, kurtosis | Characterise data distributions |
| Normality | Shapiro-Wilk test, Q-Q plot | Validate parametric test assumptions |
| Trend Analysis | OLS regression on time, t-test on slope | Test rainfall & temperature trends |
| Correlation | Pearson r with p-values | Identify climate-yield associations |
| Group Comparison | Welch's t-test | High vs low rainfall year yields |
| Regression | Simple OLS, Multiple OLS, Ridge, Lasso | Predict yield from climate variables |
| Cross-Validation | Leave-One-Out (LOO-CV) | Unbiased performance with n=12 |

---

## 📈 Key Results

| Finding | Value |
|---|---|
| Rainfall trend (1990–2025) | **+16.9 mm/year** (r=0.58, p=0.0002) ✅ |
| Temperature trend (1995–2025) | **+0.076 °C/year** (r=0.52, p=0.0025) ✅ |
| Rainfall–Yield correlation | r=-0.20 (p=0.53) — **not significant** |
| GS Rain–Yield correlation | r=-0.03 (p=0.93) — **not significant** |
| Best model (LOO R²) | Ridge Regression: 0.0499 |
| Yield range | 3.065 – 4.259 t/ha (mean: 3.649) |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install streamlit pandas numpy scikit-learn matplotlib seaborn scipy

# 2. Launch the dashboard
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📝 The Story in One Paragraph

Eldoret is getting wetter (+17 mm/year) and warmer (+0.076°C/year) — both
statistically confirmed. Yet despite this, neither total annual rainfall nor
growing-season rainfall explains maize yield variation in Uasin Gishu
(r ≈ -0.20, p > 0.05). The county receives ample rainfall in most years;
yield is not rain-constrained. Regression models trained on climate variables
alone achieve near-zero predictive power (LOO R² ≈ 0.05). The implication:
*in this high-rainfall highland environment, agronomic management —
seed variety, planting date, fertiliser timing — is the dominant yield driver.*
This is a policy-relevant, statistically honest conclusion.

---

*All data are from real observations provided by the student from the
Ministry of Agriculture and Livestock Development, Kenya.*
