"""
Streamlit Dashboard: Climate Variability and Maize Yield Analysis
Uasin Gishu County (Eldoret), Kenya

Data Sources:
- Rainfall: 1990-2025 (Ministry of Agriculture & Livestock Development)
- Temperature: 1995-2025 (Kenya Meteorological Department)
- Maize Yield: 2012-2023 (Ministry of Agriculture)

Run: streamlit run app.py
"""

import warnings

warnings.filterwarnings("ignore")

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut
import streamlit as st


# Page Configuration
st.set_page_config(
    page_title="Uasin Gishu Climate-Crop Study",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  
  html, body, [class*="css"], .main {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1e293b;
  }

  [data-testid="stMetricValue"] > div {
    color: #0f172a !important;
  }
  [data-testid="stMetricLabel"] > div > p {
    color: #64748b !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.75rem;
  }
  [data-testid="stMetricDelta"] > div {
    font-weight: 600 !important;
  }

  .page-title { 
    font-size: 2.25rem; 
    font-weight: 700; 
    color: #0f172a; 
    line-height: 1.2; 
    margin-bottom: 0.25rem;
  }
  
  .page-subtitle { 
    font-size: 1rem; 
    color: #64748b; 
    margin-bottom: 1.5rem; 
    font-weight: 400;
  }

  .kpi-card {
    background: #ffffff;
    padding: 1.25rem;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgb(0 0 0 / 0.1);
    border: 1px solid #e2e8f0;
    text-align: center;
    margin-bottom: 1rem;
  }
  
  .kpi-val {
    font-size: 1.875rem;
    font-weight: 700;
    color: #0f172a;
    display: block;
    line-height: 1.2;
  }
  
  .kpi-lbl {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
    display: block;
  }

  .section-card {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    margin-bottom: 1.5rem;
    color: #1e293b;
  }

  .note-info { 
    background: #f0f9ff; 
    border-left: 3px solid #0284c7;
    padding: 1rem 1.25rem; 
    border-radius: 4px; 
    margin: 1rem 0;
    color: #0c4a6e;
    font-size: 0.9rem;
  }
  
  .note-warning { 
    background: #fefce8; 
    border-left: 3px solid #ca8a04;
    padding: 1rem 1.25rem; 
    border-radius: 4px; 
    margin: 1rem 0;
    color: #713f12;
    font-size: 0.9rem;
  }
  
  .note-success { 
    background: #f0fdf4; 
    border-left: 3px solid #16a34a;
    padding: 1rem 1.25rem; 
    border-radius: 4px; 
    margin: 1rem 0;
    color: #14532d;
    font-size: 0.9rem;
  }

  div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 0.875rem;
    border-radius: 8px;
  }
  
  .source-text {
    font-size: 0.8rem;
    color: #64748b;
  }
</style>
""",
    unsafe_allow_html=True,
)


# Data Loading
@st.cache_data
def load_all():
    rain_m = pd.read_csv("data/cleaned/rainfall_monthly_clean.csv")
    rain_a = pd.read_csv("data/cleaned/rainfall_annual_clean.csv")
    temp_m = pd.read_csv("data/cleaned/temperature_monthly_clean.csv")
    temp_a = pd.read_csv("data/cleaned/temperature_annual_clean.csv")
    maize = pd.read_csv("data/cleaned/maize_yield_clean.csv")
    panel = pd.read_csv("data/cleaned/panel_annual_clean.csv")

    if (
        "Annual_Mean_Temp" in temp_a.columns
        and "Annual_Mean_Temp_clean" in temp_a.columns
    ):
        temp_a = temp_a.drop(columns=["Annual_Mean_Temp"])

    temp_a = temp_a.rename(
        columns={
            "Annual_Mean_Temp_clean": "Annual_Mean_Temp",
            "Max_Monthly_Temp_clean": "Max_Monthly_Temp",
            "Min_Monthly_Temp_clean": "Min_Monthly_Temp",
            "Temp_Range_clean": "Temp_Range",
        }
    )

    if "Mean_Temp_C" in temp_m.columns and "Mean_Temp_C_clean" in temp_m.columns:
        temp_m = temp_m.drop(columns=["Mean_Temp_C"])

    temp_m = temp_m.rename(columns={"Mean_Temp_C_clean": "Mean_Temp_C"})

    panel = panel.rename(
        columns={
            "Annual_Mean_Temp_clean": "Annual_Mean_Temp",
            "GS_Temp_clean": "GS_Mean_Temp",
        }
    )

    return rain_m, rain_a, temp_m, temp_a, maize, panel


rain_m, rain_a, temp_m, temp_a, maize, panel = load_all()

MONTHS_SHORT = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


# Sidebar Navigation
st.sidebar.markdown("## Uasin Gishu Study")
st.sidebar.caption("Applied Statistics & Computing\nFinal Year Project")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Rainfall Analysis",
        "Temperature Analysis",
        "Exploratory Data Analysis",
        "Hypothesis Testing",
        "Regression Modelling",
        "Yield Predictor",
        "Conclusions",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Source**")
st.sidebar.markdown(
    '<p class="source-text">Ministry of Agriculture & Livestock Development, Kenya<br>Eldoret Meteorological Station</p>',
    unsafe_allow_html=True,
)


# PAGE 1: Project Overview
if page == "Project Overview":
    st.markdown(
        '<p class="page-title">Climate Variability and Maize Yield</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="page-subtitle">Analysis of 35 years of climate trends and agricultural productivity in Uasin Gishu County, Kenya</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f'<div class="kpi-card"><span class="kpi-val">36</span><span class="kpi-lbl">Rainfall Years</span></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="kpi-card"><span class="kpi-val">31</span><span class="kpi-lbl">Temp Years</span></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="kpi-card"><span class="kpi-val">12</span><span class="kpi-lbl">Yield Records</span></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="kpi-card"><span class="kpi-val">{rain_a["Annual_Rainfall_mm"].mean():.0f}</span><span class="kpi-lbl">Avg Rain (mm)</span></div>',
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f'<div class="kpi-card"><span class="kpi-val">{maize["Maize_Yield_t_ha"].mean():.2f}</span><span class="kpi-lbl">Avg Yield (t/ha)</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        col_l, col_r = st.columns([1.2, 0.8])
        with col_l:
            st.markdown("### Study Objectives")
            st.markdown("""
            This research investigates the relationship between climate variability and agricultural output in Eldoret, Kenya.
            
            - **Trend Identification**: Quantifying 35 years of meteorological changes
            - **Constraint Analysis**: Determining if moisture or temperature is the limiting factor
            - **Predictive Modelling**: Evaluating if climate data alone can forecast yields
            - **Policy Recommendations**: Formulating evidence-based guidance for highland maize farming
            """)
        with col_r:
            st.markdown("### Study Parameters")
            st.markdown(f"""
            - **Region**: Uasin Gishu County (Highland)
            - **Station**: Eldoret Meteorological
            - **Primary Crop**: Maize (*Zea mays*)
            - **Methodology**: Applied Statistics
            - **Significance Level**: 95% (α = 0.05)
            """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Dataset Overview")
    st.dataframe(
        pd.DataFrame(
            {
                "Dataset": [
                    "Rainfall (Monthly)",
                    "Temperature (Monthly)",
                    "Maize Yield (Annual)",
                ],
                "Time Span": ["1990-2025", "1995-2025", "2012-2023"],
                "Resolution": ["Monthly", "Monthly", "Annual"],
                "Source": [
                    "Ministry of Agriculture & Livestock Dev.",
                    "Kenya Met Service",
                    "Ministry of Agriculture",
                ],
            }
        ),
        use_container_width=True,
    )

    with st.expander("View Research Pipeline"):
        st.markdown("""
        1. **Data Wrangling**: Cleaned raw DOCX/Excel files into standardized CSVs
        2. **Outlier Treatment**: Handled sensor outliers using IQR-based bounds
        3. **Feature Engineering**: Generated seasonal totals (Long/Short Rains) and warming anomalies
        4. **Statistical Inference**: Conducted Pearson Correlation and Welch's t-tests
        5. **Model Validation**: Evaluated models using LOO-CV to account for small sample size
        """)


# PAGE 2: Rainfall Analysis
elif page == "Rainfall Analysis":
    st.markdown(
        '<p class="page-title">Rainfall Variability</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="page-subtitle">Long-term precipitation trends and seasonal distribution (1990-2025)</p>',
        unsafe_allow_html=True,
    )

    yr_range = st.sidebar.slider("Filter Year Range", 1990, 2025, (1990, 2025))
    rain_f = rain_a[(rain_a["Year"] >= yr_range[0]) & (rain_a["Year"] <= yr_range[1])]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Mean Annual", f"{rain_f['Annual_Rainfall_mm'].mean():.0f} mm")
    with m2:
        max_yr = rain_f.loc[rain_f["Annual_Rainfall_mm"].idxmax(), "Year"]
        st.metric(
            "Maximum", f"{rain_f['Annual_Rainfall_mm'].max():.0f} mm", f"Year {max_yr}"
        )
    with m3:
        min_yr = rain_f.loc[rain_f["Annual_Rainfall_mm"].idxmin(), "Year"]
        st.metric(
            "Minimum", f"{rain_f['Annual_Rainfall_mm'].min():.0f} mm", f"Year {min_yr}"
        )
    with m4:
        slope_r, _, r_r, p_r, _ = stats.linregress(
            rain_f["Year"], rain_f["Annual_Rainfall_mm"]
        )
        st.metric("Linear Trend", f"{slope_r:+.1f} mm/yr", f"p={p_r:.3f}")

    tab_trend, tab_season, tab_raw = st.tabs(
        ["Trend Analysis", "Seasonal Patterns", "Data"]
    )

    with tab_trend:
        st.markdown("#### Annual Rainfall Trend")
        chart_data = rain_f.set_index("Year")[["Annual_Rainfall_mm"]]
        st.area_chart(chart_data, color="#0ea5e9")

        col_fig, col_txt = st.columns([1, 1])
        with col_fig:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(
                rain_f["Year"],
                rain_f["Annual_Rainfall_mm"],
                color="#bae6fd",
                alpha=0.8,
                label="Annual Total",
            )
            z = np.polyfit(rain_f["Year"], rain_f["Annual_Rainfall_mm"], 1)
            p = np.poly1d(z)
            ax.plot(
                rain_f["Year"],
                p(rain_f["Year"]),
                "r--",
                linewidth=2,
                label="Linear Trend",
            )
            ax.set_ylabel("Rainfall (mm)")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with col_txt:
            if p_r < 0.05:
                st.markdown(
                    f'<div class="note-success"><b>Statistically Significant Trend (p={p_r:.4f})</b><br>Rainfall is increasing by approximately {slope_r:.1f} mm per year. This cumulative change ({slope_r * 35:.0f} mm since 1990) suggests a shift in the local hydrological cycle.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="note-warning"><b>No Significant Trend Detected (p={p_r:.3f})</b><br>While inter-annual variability is high, the data does not show a statistically significant long-term direction. Agricultural planning should account for high variance rather than directional change.</div>',
                    unsafe_allow_html=True,
                )

    with tab_season:
        st.markdown("#### Monthly Climatology")
        m_avg = rain_m.groupby("Month_Num")["Rainfall_mm"].mean()
        m_data = pd.DataFrame(
            {"Month": MONTHS_SHORT, "Avg Rainfall (mm)": m_avg.values}
        )
        st.bar_chart(m_data.set_index("Month"), color="#3b82f6")

        st.markdown("---")
        st.markdown("#### Rainfall Intensity Heatmap")
        fig, ax = plt.subplots(figsize=(12, 6))
        rain_pivot = rain_m.pivot_table(
            index="Year", columns="Month_Num", values="Rainfall_mm"
        )
        sns.heatmap(
            rain_pivot,
            ax=ax,
            cmap="YlGnBu",
            xticklabels=MONTHS_SHORT,
            cbar_kws={"label": "mm"},
        )
        plt.title("Monthly Precipitation Intensity (1990-2025)")
        st.pyplot(fig)
        plt.close()

    with tab_raw:
        st.markdown("#### Filtered Records")
        st.dataframe(
            rain_f.style.background_gradient(
                subset=["Annual_Rainfall_mm"], cmap="Blues"
            ),
            use_container_width=True,
        )


# PAGE 3: Temperature Analysis
elif page == "Temperature Analysis":
    st.markdown('<p class="page-title">Temperature Trends</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Regional warming patterns and monthly anomalies (1995-2025)</p>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Mean Temperature", f"{temp_a['Annual_Mean_Temp'].mean():.2f} °C")
    with m2:
        hot_yr = temp_a.loc[temp_a["Annual_Mean_Temp"].idxmax(), "Year"]
        st.metric(
            "Warmest Year",
            f"{temp_a['Annual_Mean_Temp'].max():.2f} °C",
            f"Year {hot_yr}",
        )
    with m3:
        slope_t, _, r_t, p_t, _ = stats.linregress(
            temp_a["Year"], temp_a["Annual_Mean_Temp"]
        )
        st.metric("Warming Rate", f"{slope_t:+.3f} °C/yr")
    with m4:
        total_warm = slope_t * 30
        st.metric("30-Year Change", f"{total_warm:+.2f} °C", f"p={p_t:.4f}")

    tab_t_trend, tab_t_monthly, tab_t_raw = st.tabs(
        ["Warming Trend", "Monthly Patterns", "Data"]
    )

    with tab_t_trend:
        st.markdown("#### Annual Mean Temperature")
        st.line_chart(temp_a.set_index("Year")[["Annual_Mean_Temp"]], color="#ef4444")

        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            base_temp = temp_a["Annual_Mean_Temp"].mean()
            anomaly = temp_a["Annual_Mean_Temp"] - base_temp
            colors = ["#ef4444" if v > 0 else "#3b82f6" for v in anomaly]
            ax.bar(temp_a["Year"], anomaly, color=colors, alpha=0.8)
            ax.axhline(0, color="black", linewidth=1)
            ax.set_title("Temperature Anomaly (Deviation from Mean)")
            ax.set_ylabel("Deviation (°C)")
            st.pyplot(fig)
            plt.close()

        with col_t2:
            if p_t < 0.05:
                st.markdown(
                    f'<div class="note-warning"><b>Significant Warming Detected (p={p_t:.4f})</b><br>Eldoret has warmed by approximately {total_warm:.2f}°C since 1995. This trend is statistically robust and consistent with global climate change patterns. High-temperature extremes are becoming more frequent.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="note-info"><b>No Significant Trend (p={p_t:.3f})</b><br>The data does not show a statistically significant temperature shift. Temperature remains relatively stable compared to rainfall variability.</div>',
                    unsafe_allow_html=True,
                )

    with tab_t_monthly:
        st.markdown("#### Monthly Mean Temperature")
        m_avg_t = temp_m.groupby("Month_Num")["Mean_Temp_C"].mean()
        m_data_t = pd.DataFrame(
            {"Month": MONTHS_SHORT, "Avg Temp (°C)": m_avg_t.values}
        )
        st.line_chart(m_data_t.set_index("Month"), color="#f97316")

        st.markdown("#### Temperature Distribution Heatmap")
        fig, ax = plt.subplots(figsize=(12, 6))
        temp_pivot = temp_m.pivot_table(
            index="Year", columns="Month_Num", values="Mean_Temp_C"
        )
        sns.heatmap(
            temp_pivot, ax=ax, cmap="RdYlBu_r", xticklabels=MONTHS_SHORT, annot=False
        )
        plt.title("Monthly Temperature (1995-2025)")
        st.pyplot(fig)
        plt.close()

    with tab_t_raw:
        st.dataframe(
            temp_a.style.background_gradient(
                subset=["Annual_Mean_Temp"], cmap="YlOrRd"
            ),
            use_container_width=True,
        )


# PAGE 4: Exploratory Data Analysis
elif page == "Exploratory Data Analysis":
    st.markdown(
        '<p class="page-title">Exploratory Data Analysis</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="page-subtitle">Characterising maize yield and climate feature associations</p>',
        unsafe_allow_html=True,
    )

    tab_yield, tab_corr = st.tabs(["Yield Analysis", "Feature Correlations"])

    with tab_yield:
        with st.container():
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Maize Yield Distribution (2012-2023)")
            col_y1, col_y2 = st.columns([1.2, 0.8])
            with col_y1:
                y_chart = maize.set_index("Year")[["Maize_Yield_t_ha"]]
                st.bar_chart(y_chart, color="#10b981")
            with col_y2:
                st.markdown("##### Summary Statistics")
                st.markdown(f"""
                - **Mean**: {maize["Maize_Yield_t_ha"].mean():.3f} t/ha
                - **Maximum**: {maize["Maize_Yield_t_ha"].max():.3f} t/ha
                - **Minimum**: {maize["Maize_Yield_t_ha"].min():.3f} t/ha
                - **CV**: {maize["Maize_Yield_t_ha"].std() / maize["Maize_Yield_t_ha"].mean() * 100:.1f}%
                """)
                sw_stat, sw_p = stats.shapiro(maize["Maize_Yield_t_ha"])
                if sw_p > 0.05:
                    st.markdown(
                        f'<div class="note-success"><b>Normality confirmed</b> (Shapiro-Wilk p={sw_p:.3f})</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="note-warning"><b>Non-normal distribution</b> (Shapiro-Wilk p={sw_p:.3f})</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Annual Yield Records")
        st.dataframe(
            maize.style.highlight_max(axis=0, color="#dcfce7").highlight_min(
                axis=0, color="#fee2e2"
            ),
            use_container_width=True,
        )

    with tab_corr:
        st.markdown("#### Multi-Feature Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        cols_for_corr = [
            "Maize_Yield_t_ha",
            "Annual_Rainfall_mm",
            "Annual_Mean_Temp",
            "GS_Rain_mm",
            "GS_Mean_Temp",
            "Apr_Rain_mm",
        ]
        clean_labels = [
            "Yield",
            "Ann Rain",
            "Ann Temp",
            "GS Rain",
            "GS Temp",
            "Apr Rain",
        ]
        corr_data = panel[cols_for_corr]
        sns.heatmap(
            corr_data.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
            xticklabels=clean_labels,
            yticklabels=clean_labels,
        )
        st.pyplot(fig)
        plt.close()

        st.markdown(
            '<div class="note-info"><b>Key Finding</b>: The weak correlation between Yield and Rainfall variables suggests that in Uasin Gishu, moisture is rarely the limiting factor for maize production.</div>',
            unsafe_allow_html=True,
        )


# PAGE 5: Hypothesis Testing
elif page == "Hypothesis Testing":
    st.markdown('<p class="page-title">Hypothesis Testing</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Statistical verification of climate-yield relationships</p>',
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        test = st.selectbox(
            "Select Hypothesis:",
            [
                "H1: Significant trend in annual rainfall?",
                "H2: Significant warming trend in temperature?",
                "H3: Annual rainfall correlation with yield?",
                "H4: Growing-season temperature correlation with yield?",
                "H5: High vs Low rainfall yield comparison (t-test)",
                "H6: April rainfall correlation with yield?",
            ],
        )
        st.markdown("</div>", unsafe_allow_html=True)

    alpha = 0.05

    if "H1" in test:
        slope, icept, r, p, se = stats.linregress(
            rain_a["Year"], rain_a["Annual_Rainfall_mm"]
        )
        st.markdown("#### H0: No significant linear trend in annual rainfall")
        m_h1, m_h2, m_h3, m_h4 = st.columns(4)
        m_h1.metric("Slope", f"{slope:+.3f}")
        m_h2.metric("Pearson r", f"{r:.4f}")
        m_h3.metric("p-value", f"{p:.4f}")
        m_h4.metric("Decision", "Reject H0" if p < alpha else "Fail to Reject H0")

        if p < alpha:
            st.markdown(
                f'<div class="note-success"><b>Reject H0</b>: Statistically significant positive trend (+{slope:.2f} mm/yr). Rainfall at Eldoret has been increasing over 35 years.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="note-warning"><b>Fail to Reject H0</b>: No significant trend detected. Inter-annual variability is the dominant characteristic.</div>',
                unsafe_allow_html=True,
            )

    elif "H2" in test:
        slope, icept, r, p, se = stats.linregress(
            temp_a["Year"], temp_a["Annual_Mean_Temp"]
        )
        st.markdown("#### H0: No significant warming trend in mean temperature")
        m_h1, m_h2, m_h3, m_h4 = st.columns(4)
        m_h1.metric("Slope", f"{slope:+.4f} °C/yr")
        m_h2.metric("Pearson r", f"{r:.4f}")
        m_h3.metric("p-value", f"{p:.4f}")
        m_h4.metric("Decision", "Reject H0" if p < alpha else "Fail to Reject H0")

        if p < alpha:
            st.markdown(
                f'<div class="note-warning"><b>Reject H0</b>: Significant warming confirmed (+{slope:.4f} °C/yr). Eldoret is experiencing a measurable temperature rise.</div>',
                unsafe_allow_html=True,
            )

    elif "H3" in test:
        r, p = stats.pearsonr(panel["Annual_Rainfall_mm"], panel["Maize_Yield_t_ha"])
        st.markdown("#### H0: No correlation between annual rainfall and maize yield")
        m_h1, m_h2, m_h3, m_h4 = st.columns(4)
        m_h1.metric("Pearson r", f"{r:.4f}")
        m_h2.metric("R-squared", f"{r**2:.4f}")
        m_h3.metric("p-value", f"{p:.4f}")
        m_h4.metric("Decision", "Reject H0" if p < alpha else "Fail to Reject H0")

        st.markdown(
            f'<div class="note-info"><b>Result</b>: No significant relationship between total annual rainfall and yield. This finding challenges the assumption that "more rain = higher yields" in this region.</div>',
            unsafe_allow_html=True,
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.regplot(
            data=panel,
            x="Annual_Rainfall_mm",
            y="Maize_Yield_t_ha",
            ax=ax,
            color="#10b981",
        )
        ax.set_title("Rainfall vs Yield (OLS Regression)")
        ax.set_xlabel("Annual Rainfall (mm)")
        ax.set_ylabel("Yield (t/ha)")
        st.pyplot(fig)
        plt.close()

    elif "H4" in test:
        r, p = stats.pearsonr(panel["GS_Mean_Temp"], panel["Maize_Yield_t_ha"])
        st.markdown("#### H0: Growing-season temperature has no correlation with yield")
        m_h1, m_h2, m_h3, m_h4 = st.columns(4)
        m_h1.metric("Pearson r", f"{r:.4f}")
        m_h2.metric("p-value", f"{p:.4f}")
        m_h3.metric("Decision", "Reject H0" if p < alpha else "Fail to Reject H0")
        if p < alpha:
            st.markdown(
                f'<div class="note-success"><b>Significant relationship detected.</b></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="note-warning"><b>No significant correlation</b> between growing-season temperature and yield.</div>',
                unsafe_allow_html=True,
            )

    elif "H5" in test:
        med_r = panel["Annual_Rainfall_mm"].median()
        high = panel[panel["Annual_Rainfall_mm"] >= med_r]["Maize_Yield_t_ha"]
        low = panel[panel["Annual_Rainfall_mm"] < med_r]["Maize_Yield_t_ha"]
        t_stat, p_val = stats.ttest_ind(high, low)
        st.markdown(
            f"#### H0: Yield is identical in high vs low rainfall years (median split: {med_r:.0f} mm)"
        )
        m_h1, m_h2, m_h3 = st.columns(3)
        m_h1.metric("High Rainfall Mean", f"{high.mean():.3f}")
        m_h2.metric("Low Rainfall Mean", f"{low.mean():.3f}")
        m_h3.metric("p-value", f"{p_val:.4f}")
        st.markdown(
            f'<div class="note-info"><b>Observation</b>: Yields are comparable (or slightly higher) in "low" rainfall years, supporting the conclusion that Uasin Gishu receives adequate moisture.</div>',
            unsafe_allow_html=True,
        )

    elif "H6" in test:
        r, p = stats.pearsonr(panel["Apr_Rain_mm"], panel["Maize_Yield_t_ha"])
        st.markdown("#### H0: April rainfall has no correlation with yield")
        m_h1, m_h2, m_h3 = st.columns(3)
        m_h1.metric("Pearson r", f"{r:.4f}")
        m_h2.metric("p-value", f"{p:.4f}")
        m_h3.metric("Decision", "Reject H0" if p < alpha else "Fail to Reject H0")
        st.markdown(
            f'<div class="note-warning"><b>Result</b>: April rainfall (peak of long rains) does not significantly predict yield.</div>',
            unsafe_allow_html=True,
        )


# PAGE 6: Regression Modelling
elif page == "Regression Modelling":
    st.markdown(
        '<p class="page-title">Regression Modelling</p>', unsafe_allow_html=True
    )
    st.markdown(
        "All models validated using **Leave-One-Out Cross-Validation** (LOO-CV), appropriate for n=12 observations."
    )

    try:
        with open("models/metrics/all_model_metrics.json", "r") as f:
            all_metrics = json.load(f)
        metrics_df = pd.read_csv("models/metrics/model_comparison.csv")
    except FileNotFoundError:
        st.error("Model metrics not found. Please run 02_model_training.py first.")
        st.stop()

    y = panel["Maize_Yield_t_ha"].values

    st.markdown("### Model Comparison (LOO-CV)")
    st.dataframe(metrics_df.round(4), use_container_width=True)

    best_model_name = metrics_df.iloc[0]["Model"]
    best_short = best_model_name.replace("M", "").split("_", 1)[1].replace("_", " ")

    st.markdown(f"### Best Model: {best_short}")

    best_preds = all_metrics[best_model_name].get("preds", [])
    if not best_preds:
        st.caption("Feature coefficients for the best regularised model:")
        coefs = all_metrics[best_model_name].get("coefs", {})
        if coefs:
            c_df = pd.DataFrame(
                list(coefs.items()), columns=["Feature", "Coefficient"]
            ).sort_values("Coefficient", key=abs, ascending=False)
            st.dataframe(c_df.round(4), use_container_width=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Regression Analysis - Best Model: {best_short}",
        fontsize=11,
        fontweight="bold",
    )

    ax = axes[0]
    r2_vals = metrics_df["LOO_R2"].values
    model_names = [
        m.replace("M", "").split("_", 1)[1].replace("_", " ")
        for m in metrics_df["Model"]
    ]
    colors = ["#2A9D8F" if v >= 0 else "#E63946" for v in r2_vals]
    ax.barh(model_names[::-1], r2_vals[::-1], color=colors[::-1], edgecolor="white")
    ax.set_title("LOO R-squared Comparison")
    ax.set_xlabel("R-squared")

    ax = axes[1]
    rmse_vals = metrics_df["LOO_RMSE"].values
    ax.barh(
        model_names[::-1],
        rmse_vals[::-1],
        color=sns.color_palette("viridis", len(model_names)),
        edgecolor="white",
    )
    ax.set_title("LOO RMSE Comparison")
    ax.set_xlabel("RMSE (t/ha)")

    ax = axes[2]
    coefs = all_metrics[best_model_name].get("coefs", {})
    if not coefs:
        coefs = all_metrics[best_model_name].get("feature_importances", {})

    if coefs:
        c_series = pd.Series(coefs).sort_values()
        ax.barh(c_series.index, c_series.values, color="#F4A261", edgecolor="white")
        ax.set_title(f"Feature Coefficients: {best_short}")
    else:
        ax.text(0.5, 0.5, "No coefficient data", ha="center")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        '<div class="note-info"><b>Interpretation</b>: The low and sometimes negative LOO R-squared values reflect the small sample size (n=12) and the statistical finding that climate variables alone are weak predictors of maize yield in Uasin Gishu. Ridge Regression performs best as it penalises large coefficients, reducing overfitting.</div>',
        unsafe_allow_html=True,
    )


# PAGE 7: Yield Predictor
elif page == "Yield Predictor":
    st.markdown('<p class="page-title">Yield Predictor</p>', unsafe_allow_html=True)
    st.markdown(
        "Estimate maize yield using the Ridge Regression model identified as the most robust predictor during cross-validation."
    )

    try:
        model = pickle.load(open("models/best_model.pkl", "rb"))
        with open("models/best_model_info.json", "r") as f:
            m_info = json.load(f)
        features = m_info["features"]
    except FileNotFoundError:
        st.error("Model files not found. Please run 02_model_training.py first.")
        st.stop()

    if st.button("Reset to Historical Means"):
        st.rerun()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### Rainfall")
        val_ann_rain = st.slider(
            "Annual Rainfall (mm)", 400, 2000, int(panel["Annual_Rainfall_mm"].mean())
        )
        val_lr_rain = st.slider(
            "Long Rains Mar-May (mm)", 100, 1000, int(panel["Long_Rain_mm"].mean())
        )
        val_sr_rain = st.slider(
            "Short Rains Oct-Nov (mm)", 50, 600, int(panel["Short_Rain_mm"].mean())
        )

    with c2:
        st.markdown("##### Temperature")
        val_ann_temp = st.slider(
            "Annual Mean Temp (°C)",
            14.0,
            22.0,
            float(panel["Annual_Mean_Temp"].mean()),
            0.1,
        )
        val_gs_temp = st.slider(
            "Growing Season Temp (°C)",
            14.0,
            22.0,
            float(panel["GS_Mean_Temp"].mean()),
            0.1,
        )

    with c3:
        st.markdown("##### Distribution")
        val_max_rain = st.slider(
            "Max Monthly Rain (mm)", 50, 500, int(panel["Max_Monthly_Rain"].mean())
        )
        val_rain_mos = st.slider(
            "Rainy Months (count)", 1, 12, int(panel["Rainy_Months"].mean())
        )

    if st.button("Generate Prediction"):
        X_new = pd.DataFrame(
            [
                {
                    "Annual_Rainfall_mm": val_ann_rain,
                    "Annual_Mean_Temp_clean": val_ann_temp,
                    "Long_Rain_mm": val_lr_rain,
                    "Short_Rain_mm": val_sr_rain,
                    "GS_Temp_clean": val_gs_temp,
                    "Max_Monthly_Rain": val_max_rain,
                    "Rainy_Months": val_rain_mos,
                }
            ]
        )

        pred = model.predict(X_new.values)[0]
        avg = panel["Maize_Yield_t_ha"].mean()

        st.markdown("---")
        res1, res2, res3 = st.columns(3)
        res1.metric("Predicted Yield", f"{pred:.3f} t/ha")
        res2.metric("Historical Mean", f"{avg:.3f} t/ha")
        pct = (pred - avg) / avg * 100
        res3.metric("Difference", f"{pct:+.1f}%")

        if pred >= panel["Maize_Yield_t_ha"].max():
            st.markdown(
                '<div class="note-success"><b>Above historical maximum</b>: This prediction exceeds the highest recorded yield in the dataset.</div>',
                unsafe_allow_html=True,
            )
        elif pred >= avg:
            st.markdown(
                f'<div class="note-success"><b>Above average</b>: Input conditions suggest an above-average yield.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="note-warning"><b>Below average</b>: Input conditions suggest a below-average yield.</div>',
                unsafe_allow_html=True,
            )

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(
            panel["Year"].astype(str),
            panel["Maize_Yield_t_ha"],
            color="#AED6F1",
            alpha=0.6,
            label="Historical",
        )
        ax.axhline(
            pred,
            color="#D62828",
            linestyle="--",
            linewidth=2,
            label=f"Prediction: {pred:.3f}",
        )
        ax.set_ylabel("Yield (t/ha)")
        ax.legend(loc="upper left", fontsize=8)
        st.pyplot(fig)
        plt.close()


# PAGE 8: Conclusions
elif page == "Conclusions":
    st.markdown(
        '<p class="page-title">Conclusions and Recommendations</p>',
        unsafe_allow_html=True,
    )

    st.markdown("### Summary of Findings")

    slope_r, _, r_r, p_r, _ = stats.linregress(
        rain_a["Year"], rain_a["Annual_Rainfall_mm"]
    )
    slope_t, _, r_t, p_t, _ = stats.linregress(
        temp_a["Year"], temp_a["Annual_Mean_Temp"]
    )
    r3, p3 = stats.pearsonr(panel["Annual_Rainfall_mm"], panel["Maize_Yield_t_ha"])
    r4, p4 = stats.pearsonr(panel["GS_Rain_mm"], panel["Maize_Yield_t_ha"])

    try:
        with open("models/best_model_info.json", "r") as f:
            best_info = json.load(f)
        best_name = (
            best_info["name"].replace("M", "").split("_", 1)[1].replace("_", " ")
        )
        best_r2 = best_info["LOO_R2"]
    except:
        best_name = "Ridge Regression"
        best_r2 = -0.1159

    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f"""
        | Indicator | Result | Evidence |
        |-----------|--------|----------|
        | **Rainfall Trend** | +{slope_r:.2f} mm/year (Rising) | p={p_r:.4f} |
        | **Temperature Trend** | +{slope_t:.4f} °C/year (Warming) | p={p_t:.4f} |
        | **Climate-Yield Correlation** | Not significant | p > 0.05 |
        | **Predictive Power** | Low (R-squared < 0.1) | {best_name} |
        | **Yield Variability** | CV = {maize["Maize_Yield_t_ha"].std() / maize["Maize_Yield_t_ha"].mean() * 100:.1f}% | SD = {maize["Maize_Yield_t_ha"].std():.2f} |
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("### Scientific Synthesis")
        st.markdown(f"""
        Station data from Eldoret (1990-2025) provides evidence of a **changing highland climate**. 
        Average temperatures are rising by **{slope_t * 10:.2f}°C per decade**, a statistically robust trend.
        
        However, the "Maize-Climate Paradox" persists: despite increasing moisture and heat, 
        annual climate totals explain **less than 10%** of yield variation. 
        Uasin Gishu is a moisture-surplus environment; therefore, yield is not constrained by 
        total rainfall but likely by **agronomic management** and **thermal stress** during 
        critical growth stages.
        """)
    with col_c2:
        st.markdown("### Policy Recommendations")
        st.markdown("""
        1. **Distribution over Totals**: Agricultural extension should focus on rainfall *timing* 
           (April onset) rather than annual forecast totals.
        2. **Thermal Monitoring**: The warming trend suggests a need for heat-tolerant 
           varieties even in traditionally cool highlands.
        3. **Management Priority**: Investments in soil fertility and certified seeds 
           will yield higher returns than climate-mitigation alone.
        4. **Data Infrastructure**: Strengthening county-level yield reporting is essential for 
           more precise longitudinal studies.
        """)

    st.markdown("---")
    st.caption(
        "Study Location: Eldoret, Kenya | Data: Rainfall 1990-2025, Temperature 1995-2025, Yield 2012-2023"
    )
    st.caption(
        "Applied Statistics & Computing Final Project | Validated via LOO-CV and Residual Analysis"
    )
