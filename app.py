"""
app.py – Streamlit multi-page Data Analytics & Visualization app
         for Indian Crop Production data.

WHY Streamlit: it lets us build an interactive data app purely in Python
without writing any HTML/CSS/JS, making it easy to couple with pandas
and scikit-learn objects.
"""

import os
import json
import warnings
import urllib.request

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config – must be the very first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 India Crop Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS – dark premium theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f2027, #203a43, #2c5364);
    color: white;
}
section[data-testid="stSidebar"] * { color: white !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* Main background */
.stApp { background: #0a0a0f; }

/* Cards / containers */
.card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

h1, h2, h3 { color: #e2e8f0; }
p, li { color: #94a3b8; }

/* Tab styling */
button[data-baseweb="tab"] {
    font-weight: 600;
    color: #94a3b8 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8;
}

/* Selectbox / input */
div[data-baseweb="select"] > div {
    background: #1e293b !important;
    border-color: #334155 !important;
    color: #e2e8f0 !important;
}

div[data-baseweb="input"] > div {
    background: #1e293b !important;
    border-color: #334155 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

/* Plotly charts transparent bg */
.js-plotly-plot { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
RAW_CSV    = os.path.join("data", "crops.csv")
CLEAN_CSV  = os.path.join("data", "clean.csv")
METRICS_CSV= os.path.join("data", "metrics.csv")
GEOJSON_PATH = os.path.join("data", "india_states.geojson")
MODEL_DIR  = "models"

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,15,25,0.6)",
    font=dict(color="#94a3b8", family="Inter"),
    title_font=dict(color="#e2e8f0", size=16),
)

# ─────────────────────────────────────────────
# Helper: download India GeoJSON
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading India GeoJSON …")
def get_geojson():
    """
    WHY download GeoJSON: Plotly choropleth needs geographic boundaries
    to draw the map; India state boundaries are not bundled with Plotly.
    We download from a reliable public GitHub repo and cache locally.
    """
    if os.path.exists(GEOJSON_PATH):
        with open(GEOJSON_PATH, encoding="utf-8") as f:
            return json.load(f)

    urls = [
        "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson",
        "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States",
        "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json",  # fallback
    ]
    # Primary source: datameet India states GeoJSON
    primary = "https://raw.githubusercontent.com/answerquest/leaflet-example-IndiaMap/master/india_states.geojson"
    try:
        req = urllib.request.Request(primary, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        with open(GEOJSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data
    except Exception:
        pass

    # Try backup
    backup = "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson"
    try:
        req = urllib.request.Request(backup, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        with open(GEOJSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data
    except Exception:
        return None

# ─────────────────────────────────────────────
# Helper: load data
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading raw data …")
def load_raw():
    return pd.read_csv(RAW_CSV)

@st.cache_data(show_spinner="Loading clean data …")
def load_clean():
    if not os.path.exists(CLEAN_CSV):
        st.error("clean.csv not found – please run `python train.py` first.")
        st.stop()
    return pd.read_csv(CLEAN_CSV)

@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_CSV):
        return None
    return pd.read_csv(METRICS_CSV)

@st.cache_resource
def load_models():
    """
    WHY cache_resource (not cache_data): models are large objects that should
    not be re-loaded or serialised on every re-run; resource cache keeps them
    in memory across sessions.
    """
    lr  = joblib.load(os.path.join(MODEL_DIR, "linear_regression.pkl"))
    rf  = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    meta = joblib.load(os.path.join(MODEL_DIR, "meta.pkl"))
    return lr, rf, meta

# ─────────────────────────────────────────────
# State-name normalisation map for GeoJSON matching
# ─────────────────────────────────────────────
# WHY this map: GeoJSON uses specific spellings that differ from the dataset.
# We build the mapping programmatically (lowercase, replace '&' with 'and')
# plus manual overrides for known name changes (Odisha->Orissa, Uttarakhand->Uttaranchal).
def norm_state(name):
    # WHY lowercase and replace '&': standardizes naming for comparison
    normalized = name.lower().replace('&', 'and')
    return normalized

# WHY manual map: handles historical name changes not caught by programmatic normalization
MANUAL_STATE_MAP = {
    "Odisha": "Orissa",
    "Uttarakhand": "Uttaranchal",
}

# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 India Crop Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["📊 Overview", "🔍 EDA", "🤖 Model Comparison", "🔮 Predict"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Data: Indian Crop Production (1997–2015)")
    st.caption("Models: Linear Regression | Random Forest")

# ═══════════════════════════════════════════════════════
# PAGE 1: Overview
# ═══════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 📊 Dataset Overview")
    st.markdown("A bird's-eye view of the raw and cleaned Indian Crop Production data.")

    raw = load_raw()

    # WHY show both raw and clean counts: users need to see how much data was
    # removed during cleaning so they can judge the cleaning decisions.
    models_exist = os.path.exists(os.path.join(MODEL_DIR, "linear_regression.pkl"))
    if models_exist:
        clean = load_clean()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Rows",   f"{len(raw):,}")
        c2.metric("Clean Rows", f"{len(clean):,}",
                  delta=f"-{len(raw)-len(clean):,}",
                  delta_color="inverse")
        c3.metric("States",  raw["State_Name"].nunique())
        c4.metric("Crops",   raw["Crop"].nunique())
    else:
        st.warning("⚠️ Run `python train.py` to generate cleaned data and models.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw Rows",  f"{len(raw):,}")
        c2.metric("States",    raw["State_Name"].nunique())
        c3.metric("Crops",     raw["Crop"].nunique())

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Raw Data Sample")
        st.dataframe(raw.head(10), width='stretch')

    with col2:
        st.markdown("### ❓ Missing Values (Raw)")
        miss = raw.isnull().sum().reset_index()
        miss.columns = ["Column", "Missing"]
        miss["% Missing"] = (miss["Missing"] / len(raw) * 100).round(2)

        fig = px.bar(
            miss[miss["Missing"] > 0],
            x="Column", y="Missing",
            color="% Missing",
            color_continuous_scale="Reds",
            title="Missing Values by Column",
        )
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, width='stretch')

    st.markdown("### 📈 Raw Data Statistics")
    st.dataframe(raw.describe().T.style.format("{:.2f}"), width='stretch')

    if models_exist:
        st.markdown("### ✅ Cleaned Data Sample")
        st.dataframe(clean.head(10), width='stretch')
        st.markdown("### 📈 Clean Data Statistics")
        st.dataframe(clean.describe().T.style.format("{:.2f}"), width='stretch')

# ═══════════════════════════════════════════════════════
# PAGE 2: EDA
# ═══════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.markdown("# 🔍 Exploratory Data Analysis")

    if not os.path.exists(CLEAN_CSV):
        st.error("Please run `python train.py` first to generate clean.csv.")
        st.stop()

    df = load_clean()

    # ── Sidebar filters ──────────────────────────────
    # WHY filters in sidebar: let analysts drill down to a subset of interest
    # without cluttering the main canvas.
    with st.sidebar:
        st.markdown("### 🎛️ Filters")
        all_states  = ["All"] + sorted(df["State_Name"].unique().tolist())
        all_crops   = ["All"] + sorted(df["Crop"].unique().tolist())
        all_seasons = ["All"] + sorted(df["Season"].unique().tolist())

        sel_state  = st.selectbox("State",  all_states)
        sel_crop   = st.selectbox("Crop",   all_crops)
        sel_season = st.selectbox("Season", all_seasons)

    # Apply filters
    fdf = df.copy()
    if sel_state  != "All": fdf = fdf[fdf["State_Name"] == sel_state]
    if sel_crop   != "All": fdf = fdf[fdf["Crop"]       == sel_crop]
    if sel_season != "All": fdf = fdf[fdf["Season"]     == sel_season]

    st.caption(f"Showing **{len(fdf):,}** records after filters.")

    if len(fdf) == 0:
        st.warning("No data matches the selected filters.")
        st.stop()

    # ── Row 1: Histogram + Box Plot ──────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Yield Distribution")
        # WHY histogram: see the shape/skewness of yield; important for
        # understanding whether linear models are appropriate.
        fig = px.histogram(
            fdf, x="Yield", nbins=60,
            color_discrete_sequence=["#38bdf8"],
            title="Yield Histogram",
            marginal="box",
        )
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("#### Yield by Season")
        # WHY box plot: compare yield distributions across seasons; median,
        # spread, and outliers are all visible at once.
        fig = px.box(
            fdf, x="Season", y="Yield",
            color="Season",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Yield Distribution by Season",
        )
        fig.update_layout(**PLOTLY_THEME, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    # ── Row 2: Correlation Heatmap + Top 10 States ───
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Correlation Heatmap")
        # WHY heatmap: reveals linear relationships between numeric variables;
        # useful for detecting multicollinearity before modelling.
        num_cols = fdf.select_dtypes("number").drop(columns=["Crop_Year"], errors="ignore")
        corr = num_cols.corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_alpha(0)
        ax.set_facecolor("#0f172a")
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax,
            annot_kws={"color": "white"},
        )
        ax.tick_params(colors="white")
        plt.title("Numeric Correlations", color="white")
        st.pyplot(fig, width='stretch')
        plt.close()

    with col4:
        st.markdown("#### Top 10 States by Avg Yield")
        # WHY bar chart of top states: immediately shows which states are
        # the most productive, guiding policy or further analysis.
        top_states = (
            fdf.groupby("State_Name")["Yield"]
            .mean()
            .nlargest(10)
            .reset_index()
            .rename(columns={"Yield": "Avg Yield"})
        )
        fig = px.bar(
            top_states, x="Avg Yield", y="State_Name",
            orientation="h",
            color="Avg Yield",
            color_continuous_scale="Viridis",
            title="Top 10 States – Average Yield",
        )
        fig.update_layout(**PLOTLY_THEME, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width='stretch')

    # ── Row 3: Yield Trend by Year ───────────────────
    st.markdown("#### Yield Trend by Year")
    # WHY line chart: captures whether national average yield is rising or
    # falling over the 19-year period covered by the dataset.
    trend = fdf.groupby("Crop_Year")["Yield"].mean().reset_index()
    fig = px.line(
        trend, x="Crop_Year", y="Yield",
        markers=True,
        color_discrete_sequence=["#34d399"],
        title="Average Yield per Year",
    )
    fig.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig, width='stretch')

    # ── Row 4: Choropleth ────────────────────────────
    st.markdown("#### 🗺️ Avg Yield by State (Choropleth)")
    geojson = get_geojson()

    state_yield = (
        df.groupby("State_Name")["Yield"]       # use full df, not filtered
        .mean().reset_index()
        .rename(columns={"Yield": "Avg_Yield"})
    )
    
    if geojson is None:
        st.warning("Could not load India GeoJSON. Showing fallback bar chart instead.")
        fig = px.bar(
            state_yield.sort_values("Avg_Yield", ascending=False),
            x="State_Name", y="Avg_Yield",
            color="Avg_Yield", color_continuous_scale="Greens",
            title="Avg Yield by State (bar fallback)",
        )
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, width='stretch')
    else:
        # Detect the key in the GeoJSON features that holds the state name
        try:
            sample_props = geojson["features"][0]["properties"]
            geo_key = [k for k in sample_props if "state" in k.lower() or "name" in k.lower()][0]
        except (IndexError, KeyError):
            geo_key = list(geojson["features"][0]["properties"].keys())[0]
        
        # Print actual GeoJSON state names for debugging
        with st.expander("🔍 GeoJSON State Names"):
            geo_names_list = sorted([f["properties"].get(geo_key, "") for f in geojson["features"]])
            st.write(f"GeoJSON key: `{geo_key}`")
            st.write(f"Total GeoJSON states: {len(geo_names_list)}")
            st.write(geo_names_list)
        
        # Build mapping programmatically
        geo_names_lower = {f["properties"].get(geo_key, "").lower(): f["properties"].get(geo_key, "") 
                          for f in geojson["features"]}
        
        def build_state_mapping(dataset_name):
            # WHY programmatic normalization: lowercase and replace '&' with 'and'
            norm = dataset_name.lower().replace('&', 'and')
            # WHY manual map: handles historical name changes
            if dataset_name in MANUAL_STATE_MAP:
                return MANUAL_STATE_MAP[dataset_name]
            if norm in geo_names_lower:
                return geo_names_lower[norm]
            return dataset_name
        
        state_yield["State_GEO"] = state_yield["State_Name"].apply(build_state_mapping)
        
        # Calculate set differences
        data_states_set = set(state_yield["State_Name"].tolist())
        geo_states_set = {f["properties"].get(geo_key, "") for f in geojson["features"]}
        
        with st.expander("🔍 State Matching Details"):
            st.write(f"Dataset states only: {sorted(data_states_set - geo_states_set)}")
            st.write(f"GeoJSON states only: {sorted(geo_states_set - data_states_set)}")
            
            # Check which dataset states matched after mapping
            matched_geo = set(state_yield["State_GEO"].tolist())
            unmatched = state_yield[~state_yield["State_GEO"].isin(geo_states_set)]["State_Name"].tolist()
            st.write(f"Unmatched after mapping: {unmatched}")
            
            # Check for states with no polygon (in GeoJSON but not in dataset)
            no_data = geo_states_set - set(state_yield["State_GEO"].tolist())
            st.write(f"States with no data (polygon exists but no dataset match): {sorted(no_data)}")
        
        # Count states actually drawn with non-null values
        drawn_states = state_yield[state_yield["State_GEO"].isin(geo_states_set)]["State_Name"].tolist()
        st.info(f"📊 **{len(drawn_states)} states drawn with non-null yield values**")
        
        fig = px.choropleth(
            state_yield,
            geojson=geojson,
            locations="State_GEO",
            featureidkey=f"properties.{geo_key}",
            color="Avg_Yield",
            color_continuous_scale="YlGn",
            title="Average Crop Yield by State",
            hover_name="State_Name",
            hover_data={"Avg_Yield": ":.2f"},
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(**PLOTLY_THEME, height=550)
        st.plotly_chart(fig, width='stretch')

# ═══════════════════════════════════════════════════════
# PAGE 3: Model Comparison
# ═══════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown("# 🤖 Model Comparison")
    st.markdown("Comparing Linear Regression vs Random Forest on the test set.")

    metrics = load_metrics()
    if metrics is None:
        st.error("Please run `python train.py` first.")
        st.stop()

    # ── Metrics Table ────────────────────────────────
    st.markdown("### 📋 Performance Metrics")
    st.dataframe(
        metrics.style.highlight_min(subset=["RMSE", "MAE"], color="#0f3460")
                     .highlight_max(subset=["Train_R2", "Test_R2", "CV_R2_Mean"], color="#064e3b")
                     .format("{:.4f}", subset=["Train_R2","Test_R2","RMSE","MAE","CV_R2_Mean","CV_R2_Std"]),
        width='stretch',
    )

    # ── Bar charts for RMSE and R² ───────────────────
    col1, col2 = st.columns(2)

    with col1:
        # WHY RMSE chart: lower is better; easy visual comparison.
        fig = px.bar(
            metrics, x="Model", y="RMSE",
            color="Model",
            color_discrete_sequence=["#38bdf8", "#f472b6"],
            title="RMSE (lower = better)",
            text_auto=".4f",
        )
        fig.update_layout(**PLOTLY_THEME, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with col2:
        # WHY R² chart: higher is better; shows variance explained by model.
        fig = px.bar(
            metrics, x="Model", y="Test_R2",
            color="Model",
            color_discrete_sequence=["#38bdf8", "#f472b6"],
            title="Test R² (higher = better)",
            text_auto=".4f",
        )
        fig.update_layout(**PLOTLY_THEME, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    # ── Actual vs Predicted scatter plot ─────────────
    st.markdown("### 🎯 Actual vs Predicted (Random Forest)")
    # WHY this scatter: a model predicting perfectly would put all points
    # on the diagonal line y=x; the scatter shows systematic bias.
    preds_path = os.path.join(MODEL_DIR, "random_forest_preds.pkl")
    if os.path.exists(preds_path):
        y_test_arr, y_pred_arr = joblib.load(preds_path)
        # Sample for speed
        idx = np.random.choice(len(y_test_arr), min(3000, len(y_test_arr)), replace=False)
        scatter_df = pd.DataFrame({
            "Actual":    y_test_arr[idx],
            "Predicted": y_pred_arr[idx],
        })
        fig = px.scatter(
            scatter_df, x="Actual", y="Predicted",
            opacity=0.5,
            color_discrete_sequence=["#34d399"],
            title="Actual vs Predicted Yield",
        )
        # Add perfect-prediction line
        mn = scatter_df.min().min()
        mx = scatter_df.max().max()
        fig.add_shape(
            type="line", x0=mn, y0=mn, x1=mx, y1=mx,
            line=dict(color="#f472b6", width=2, dash="dash"),
        )
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, width='stretch')

    # ── Random Forest Feature Importance ─────────────
    st.markdown("### 🌳 Random Forest Feature Importance")
    # WHY feature importance: tells us which inputs the Random Forest relies
    # on most; useful for domain experts to validate model logic.
    try:
        rf_pipe = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
        prep    = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))

        # Fit prep on clean data to extract feature names
        clean = load_clean()
        FEATURES_CAT = ["State_Name", "Crop", "Season"]
        FEATURES_NUM = ["Area", "Crop_Year"]
        prep.fit(clean[FEATURES_CAT + FEATURES_NUM])

        ohe_names = prep.named_transformers_["cat"]["ohe"].get_feature_names_out(FEATURES_CAT)
        feat_names = list(ohe_names) + FEATURES_NUM

        importances = rf_pipe.named_steps["model"].feature_importances_
        feat_df = (
            pd.DataFrame({"Feature": feat_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(20)
        )

        fig = px.bar(
            feat_df, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Plasma",
            title="Top 20 Feature Importances (Random Forest)",
        )
        fig.update_layout(**PLOTLY_THEME, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width='stretch')

    except Exception as ex:
        st.warning(f"Could not render feature importance: {ex}")

# ═══════════════════════════════════════════════════════
# PAGE 4: Predict
# ═══════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown("# 🔮 Predict Crop Yield")
    st.markdown("Fill in the form below and both models will estimate the yield.")

    if not os.path.exists(os.path.join(MODEL_DIR, "linear_regression.pkl")):
        st.error("Please run `python train.py` first.")
        st.stop()

    lr, rf, meta = load_models()

    with st.form("predict_form"):
        st.markdown("### Input Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            state = st.selectbox("State", meta["states"])
        with col2:
            crop = st.selectbox("Crop", meta["crops"])
        with col3:
            season = st.selectbox("Season", meta["seasons"])

        col4, col5 = st.columns(2)
        with col4:
            # WHY 1.0 default for area: a sensible starting unit (1 hectare)
            area = st.number_input(
                "Area (hectares)", min_value=0.01, max_value=500000.0, value=100.0, step=1.0
            )
        with col5:
            year = st.number_input(
                "Crop Year",
                min_value=meta["year_min"],
                max_value=meta["year_max"] + 10,
                value=meta["year_max"],
                step=1,
            )

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        # WHY build input as a DataFrame: scikit-learn pipelines expect a
        # DataFrame with named columns matching what the pipeline was trained on.
        input_df = pd.DataFrame([{
            "State_Name": state,
            "Crop":       crop,
            "Season":     season,
            "Area":       area,
            "Crop_Year":  year,
        }])

        try:
            # WHY check for log transform flag: models trained on log1p(Yield) need expm1 conversion
            if meta.get("log_transform", False):
                lr_pred = float(np.expm1(lr.predict(input_df)[0]))
                rf_pred = float(np.expm1(rf.predict(input_df)[0]))
            else:
                lr_pred = float(lr.predict(input_df)[0])
                rf_pred = float(rf.predict(input_df)[0])

            st.markdown("---")
            st.markdown("### 📊 Prediction Results")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                    <div class="card" style="text-align:center;">
                        <h3 style="color:#38bdf8;">📈 Linear Regression</h3>
                        <h1 style="color:#e2e8f0;font-size:3rem;">{lr_pred:,.2f}</h1>
                        <p style="color:#94a3b8;">tonnes / hectare</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="card" style="text-align:center;">
                        <h3 style="color:#f472b6;">🌳 Random Forest</h3>
                        <h1 style="color:#e2e8f0;font-size:3rem;">{rf_pred:,.2f}</h1>
                        <p style="color:#94a3b8;">tonnes / hectare</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # WHY show a gauge: gives immediate visual intuition of where
            # the predicted value sits relative to min/max range.
            clean = load_clean()
            y_min = clean["Yield"].min()
            y_max = clean["Yield"].max()

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rf_pred,
                title={"text": "RF Predicted Yield (t/ha)", "font": {"color": "#e2e8f0"}},
                gauge={
                    "axis": {"range": [y_min, y_max], "tickcolor": "#94a3b8"},
                    "bar": {"color": "#f472b6"},
                    "bgcolor": "#1e293b",
                    "steps": [
                        {"range": [y_min, y_max * 0.33], "color": "#0f3460"},
                        {"range": [y_max * 0.33, y_max * 0.66], "color": "#1e4d6b"},
                        {"range": [y_max * 0.66, y_max], "color": "#155e75"},
                    ],
                },
                number={"font": {"color": "#e2e8f0"}},
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"))
            st.plotly_chart(fig, width='stretch')

        except Exception as ex:
            st.error(f"Prediction failed: {ex}")
