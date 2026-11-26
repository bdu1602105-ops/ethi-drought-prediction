"""
Drought Intelligence - Ethiopia
Agricultural Drought Risk Prediction and Recommendation System (Streamlit dashboard)

Run locally:   streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st

import train_model as tm

BASE_DIR = Path(__file__).resolve().parent
CLASSES = tm.CLASSES
RISK_COLORS = {"No Drought": "#2e8b57", "Moderate Drought": "#f0a030", "Severe Drought": "#c0392b"}
RISK_SOFT = {"No Drought": "#e4f4ea", "Moderate Drought": "#fdf0d8", "Severe Drought": "#fbe3df"}
GREEN, DARK = "#1f6b45", "#12331f"

# Approximate site coordinates (latitude, longitude) used only for the map.
COORDS = {
    "Dolo Ado": (4.17, 42.05), "Arsi Negele": (7.35, 38.70), "Adigrat": (14.28, 39.46),
    "Sinana": (7.12, 40.22), "Melka Werer": (9.33, 40.17), "Bahir Dar": (11.60, 37.38),
    "Humera": (14.28, 36.62), "Semera": (11.79, 41.01), "Nekemte": (9.09, 36.55),
    "Adaba": (7.00, 39.40), "Jijiga": (9.35, 42.80), "Pawe": (11.32, 36.40),
    "Konso": (5.25, 37.48), "Jimma": (7.67, 36.83), "Dubti": (11.73, 41.08),
    "Hossana": (7.55, 37.85), "Woldia": (11.83, 39.60), "Assosa": (10.07, 34.53),
    "Wolaita Sodo": (6.85, 37.75), "Kobo": (12.15, 39.63), "Hawassa": (7.06, 38.48),
    "Kulumsa": (8.02, 39.15), "Metema": (12.95, 36.20), "Chencha": (6.25, 37.57),
    "Gode": (5.95, 43.45), "Arba Minch": (6.03, 37.55), "Mizan Teferi": (6.99, 35.58),
    "Mekelle": (13.50, 39.47), "Debre Birhan": (9.68, 39.53), "Gewane": (10.17, 40.65),
    "Gambella": (8.25, 34.59), "Alamata": (12.42, 39.55), "Dire Dawa": (9.60, 41.87),
    "Ankober": (9.55, 39.72), "Holeta": (9.06, 38.50), "Debre Tabor": (11.85, 38.02),
    "Negele Borana": (5.33, 39.58), "Injibara": (10.95, 36.93), "Babile": (9.22, 42.32),
    "Gondar": (12.60, 37.47), "Bako": (9.12, 37.05), "Axum": (14.12, 38.72),
    "Debre Markos": (10.33, 37.72), "Bale Robe": (7.12, 40.00), "Jinka": (5.78, 36.57),
}

LABELS = {
    "Rainfall_mm": "Rainfall (mm)",
    "Max_Temp_C": "Max temperature (°C)",
    "Min_Temp_C": "Min temperature (°C)",
    "Humidity_pct": "Humidity (%)",
    "Evapotranspiration_mm": "Evapotranspiration (mm/day)",
    "Soil_Moisture_pct": "Soil moisture (%)",
    "Mean_Temp_C": "Mean temperature (°C)",
    "Temp_Range_C": "Temperature range (°C)",
    "Rain_ET_Ratio": "Rainfall / evapotranspiration",
}

st.set_page_config(page_title="Drought Intelligence · Ethiopia", page_icon="🌾", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", system-ui, sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3 { color: #12331f; letter-spacing: -0.01em; }
.hero { background: linear-gradient(120deg, #12331f 0%, #1f6b45 60%, #3f9462 100%); color: #ffffff;
        padding: 1.5rem 2rem; border-radius: 18px; margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(18,51,31,.18); }
.hero h1 { color: #ffffff !important; font-size: 1.85rem; margin: 0 0 .25rem 0; line-height: 1.2; }
.hero p { color: #dff0e5; margin: 0; font-size: 1rem; }
.kpi { background: #ffffff; border: 1px solid #e3ebe5; border-radius: 14px; padding: .9rem 1.1rem;
       box-shadow: 0 2px 8px rgba(18,51,31,.05); height: 100%; }
.kpi .label { color: #5b6f62; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }
.kpi .value { color: #12331f; font-size: 1.7rem; font-weight: 750; line-height: 1.25; }
.kpi .sub { color: #7a8c80; font-size: .8rem; }
.card { background: #ffffff; border: 1px solid #e3ebe5; border-radius: 16px; padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 10px rgba(18,51,31,.05); margin-bottom: 1rem; }
.result { border-radius: 18px; padding: 1.3rem 1.6rem; margin-bottom: 1rem; border: 1px solid rgba(0,0,0,.06); }
.result .tag { font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; color: #4b5e52; }
.result .big { font-size: 2.1rem; font-weight: 800; line-height: 1.15; }
.result .small { color: #3d4f44; font-size: .95rem; margin-top: .2rem; }
.chip { display: inline-block; background: #e8f3ec; color: #12331f; border: 1px solid #cfe5d6;
        padding: .32rem .8rem; border-radius: 999px; margin: 0 .4rem .4rem 0; font-weight: 600; font-size: .9rem; }
.chip.main { background: #1f6b45; color: #ffffff; border-color: #1f6b45; }
.callout { border-left: 5px solid #1f6b45; background: #eef6f1; color: #12331f; padding: .9rem 1.1rem;
           border-radius: 10px; margin: .6rem 0 1rem 0; font-size: .95rem; }
.callout.warn { border-left-color: #f0a030; background: #fff6e6; }
.callout b { color: #12331f; }
.pill { display: inline-block; padding: .18rem .65rem; border-radius: 999px; font-size: .78rem; font-weight: 700; }
.section-note { color: #5b6f62; font-size: .9rem; margin-top: -.4rem; margin-bottom: .6rem; }
.action li { margin-bottom: .35rem; color: #1d3a29; }
[data-testid="stSidebar"] .brand { font-size: 1.35rem; font-weight: 800; color: #ffffff; margin-bottom: .1rem; }
[data-testid="stSidebar"] .brand-sub { color: #a9cdb6; font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; }
[data-testid="stSidebar"] .status { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14);
        border-radius: 12px; padding: .7rem .9rem; color: #e8f1ec; font-size: .85rem; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.15); }
[data-testid="stMetricValue"] { color: #12331f; }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Data + model loading (auto-trains if the saved files are missing or incompatible)
# --------------------------------------------------------------------------------------
def _find_source_data() -> Path:
    for p in [tm.DATA_PATH, BASE_DIR / "data" / "Ethiopia_Drought_Risk_Dataset.xlsx",
              BASE_DIR / "Ethiopia_Drought_Risk_Dataset.xlsx"]:
        if p.exists():
            return p
    st.error("Dataset not found. Put `data/drought_data.csv` or the Excel file next to app.py.")
    st.stop()


@st.cache_resource(show_spinner="Preparing the drought model…")
def load_artifacts():
    """Load the saved model; if it is missing or cannot run here, retrain a fresh one automatically."""
    try:
        meta = json.loads(tm.META_PATH.read_text())
        model = joblib.load(tm.MODEL_PATH)
        probe = pd.DataFrame([{f: meta["feature_ranges"][f]["median"] for f in tm.RAW_FEATURES}])
        model.predict_proba(tm.build_features(probe))          # make sure it really works in this environment
    except Exception:
        meta = tm.run(_find_source_data(), quick=True, verbose=False)
        model = joblib.load(tm.MODEL_PATH)
    return model, meta


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not tm.DATA_PATH.exists():
        tm.run(_find_source_data(), quick=True, verbose=False)
    df = pd.read_csv(tm.DATA_PATH)
    df["Drought_Risk_Category"] = pd.Categorical(df["Drought_Risk_Category"], CLASSES, ordered=True)
    df["Is_Drought"] = (df["Drought_Risk_Category"] != "No Drought").astype(int)
    return df


model, meta = load_artifacts()
df = load_data()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def predict_proba(values: dict) -> pd.Series:
    X = tm.build_features(pd.DataFrame([values]))
    names = [CLASSES[c] if isinstance(c, (int, np.integer)) else c for c in model.classes_]   # 0/1/2 or text labels
    p = pd.Series(model.predict_proba(X)[0], index=names)
    return p.reindex(CLASSES).fillna(0.0)


def risk_index(p: pd.Series) -> float:
    return float(100 * (0.5 * p["Moderate Drought"] + 1.0 * p["Severe Drought"]))


def kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def style(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, Segoe UI, sans-serif", color="#1d3a29", size=13),
        title=dict(font=dict(size=15, color="#12331f"), x=0.0), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
    )
    fig.update_xaxes(gridcolor="#e8eee9", zerolinecolor="#e8eee9")
    fig.update_yaxes(gridcolor="#e8eee9", zerolinecolor="#e8eee9")
    return fig


def show(fig: go.Figure, **kw):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, **kw)


def callout(html: str, warn: bool = False):
    st.markdown(f'<div class="callout{" warn" if warn else ""}">{html}</div>', unsafe_allow_html=True)


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


@st.cache_data(show_spinner=False)
def location_table(_df: pd.DataFrame) -> pd.DataFrame:
    g = _df.groupby("Location")
    out = g.agg(Region=("Region", "first"), Zone=("Agro_Ecological_Zone", lambda s: s.mode().iat[0]),
                Observations=("Is_Drought", "size"), Drought_Share=("Is_Drought", "mean"),
                Rainfall=("Rainfall_mm", "median"), MaxTemp=("Max_Temp_C", "median"),
                SoilMoisture=("Soil_Moisture_pct", "median")).reset_index()
    sev = _df[_df.Drought_Risk_Category == "Severe Drought"].groupby("Location").size()
    out["Severe_Share"] = out["Location"].map(sev).fillna(0) / out["Observations"]
    out["lat"] = out["Location"].map(lambda l: COORDS.get(l, (np.nan, np.nan))[0])
    out["lon"] = out["Location"].map(lambda l: COORDS.get(l, (np.nan, np.nan))[1])
    return out


def recommend_crops(risk: str, zone: str, top: int = 5):
    sub = df[(df.Drought_Risk_Category == risk) & (df.Agro_Ecological_Zone == zone)]
    basis = f"{zone} zone"
    if len(sub) < 8:
        sub = df[df.Drought_Risk_Category == risk]
        basis = "all zones"
    crops = sub["Recommended_Crops"].str.split(", ").explode().str.strip()
    share = (crops.value_counts() / len(sub)).head(top)
    return share, basis, len(sub)


def action_list(risk: str) -> list[str]:
    text = df[df.Drought_Risk_Category == risk]["Recommended_Actions"].mode().iat[0]
    return [s.strip().rstrip(".") + "." for s in text.split(". ") if s.strip()]


locs = location_table(df)

# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
PAGES = ["Overview", "Drought Prediction", "Regional Analysis", "Model Insights", "About"]
with st.sidebar:
    st.markdown('<div class="brand">🌾 Drought Intelligence</div>'
                '<div class="brand-sub">Ethiopia · Agricultural decision support</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    hold = meta["holdout"]
    st.markdown(
        f'<div class="status"><b>Model ready</b><br>{meta["model_name"]}<br>'
        f'Test accuracy: {pct(hold["accuracy"])}<br>{meta["n_rows"]:,} records · {df.Location.nunique()} sites</div>',
        unsafe_allow_html=True)


def hero(title: str, subtitle: str):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# 1. OVERVIEW
# --------------------------------------------------------------------------------------
if page == "Overview":
    hero("Agricultural Drought Risk in Ethiopia",
         "Monitor drought risk across regions, test climate scenarios and get crop and action recommendations.")

    drought_share = df.Is_Drought.mean()
    severe_share = (df.Drought_Risk_Category == "Severe Drought").mean()
    reg = df.groupby("Region").Is_Drought.mean().sort_values(ascending=False)
    c = st.columns(5)
    c[0].markdown(kpi("Monitoring sites", f"{df.Location.nunique()}", f"{df.Region.nunique()} regions"), unsafe_allow_html=True)
    c[1].markdown(kpi("Observations", f"{len(df):,}", "climate records"), unsafe_allow_html=True)
    c[2].markdown(kpi("Drought share", pct(drought_share), "moderate + severe"), unsafe_allow_html=True)
    c[3].markdown(kpi("Severe drought", pct(severe_share), "of all records"), unsafe_allow_html=True)
    c[4].markdown(kpi("Highest-risk region", reg.index[0], f"{pct(reg.iloc[0], 0)} of records in drought"),
                  unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1.7, 1])
    with left:
        m = locs.dropna(subset=["lat"]).copy()
        m["Drought %"] = (m["Drought_Share"] * 100).round(1)
        fig = px.scatter_map(
            m, lat="lat", lon="lon", color="Drought %", size=[16] * len(m), hover_name="Location",
            hover_data={"Region": True, "Zone": True, "Drought %": True, "Rainfall": ":.0f", "lat": False, "lon": False},
            color_continuous_scale=["#2e8b57", "#f4d35e", "#e8833a", "#c0392b"], range_color=[0, 100],
            zoom=4.6, center={"lat": 9.0, "lon": 39.6}, map_style="carto-positron",
            title="Share of records in drought, by monitoring site")
        style(fig, 470, legend=False)
        fig.update_layout(coloraxis_colorbar=dict(title="% drought", thickness=12, len=0.7))
        show(fig)
        st.caption("Site positions are approximate and used for visualisation only.")
    with right:
        counts = df.Drought_Risk_Category.value_counts().reindex(CLASSES)
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.62, sort=False,
                               marker=dict(colors=[RISK_COLORS[k] for k in counts.index]),
                               textinfo="percent", hovertemplate="%{label}: %{value} records<extra></extra>"))
        style(fig, 300)
        fig.update_layout(title="Risk category mix", annotations=[dict(
            text=f"<b>{pct(drought_share, 0)}</b><br>in drought", x=0.5, y=0.5, showarrow=False, font=dict(size=16))])
        show(fig)
        top = locs.sort_values("Drought_Share", ascending=False).head(3)
        callout("<b>Hotspots:</b> " + ", ".join(f"{r.Location} ({pct(r.Drought_Share, 0)})" for r in top.itertuples())
                + " have the highest share of drought records.")

    a, b = st.columns(2)
    with a:
        t = df.groupby(["Region", "Drought_Risk_Category"], observed=True).size().unstack(fill_value=0)
        t = t.reindex(columns=CLASSES, fill_value=0)
        t = (t.div(t.sum(1), axis=0) * 100).loc[reg.index[::-1]]
        fig = go.Figure([go.Bar(y=t.index, x=t[k], name=k, orientation="h", marker_color=RISK_COLORS[k],
                                hovertemplate="%{y}: %{x:.1f}%<extra>" + k + "</extra>") for k in CLASSES])
        fig.update_layout(barmode="stack", xaxis_title="% of records")
        show(style(fig, 400).update_layout(title="Risk mix by region"))
    with b:
        z = df.groupby("Agro_Ecological_Zone").agg(Drought=("Is_Drought", "mean"), Rain=("Rainfall_mm", "median"),
                                                   Soil=("Soil_Moisture_pct", "median")).reset_index()
        z = z.sort_values("Drought", ascending=False)
        fig = go.Figure(go.Bar(x=z["Agro_Ecological_Zone"], y=z["Drought"] * 100, marker_color=GREEN,
                               text=[f"{v:.0f}%" for v in z["Drought"] * 100], textposition="outside",
                               customdata=z[["Rain", "Soil"]],
                               hovertemplate="%{x}<br>Drought: %{y:.1f}%<br>Median rainfall: %{customdata[0]:.0f} mm"
                                             "<br>Median soil moisture: %{customdata[1]:.1f}%<extra></extra>"))
        fig.update_layout(yaxis_title="% of records in drought", yaxis_range=[0, max(10, z["Drought"].max() * 120)])
        show(style(fig, 400, legend=False).update_layout(title="Drought share by agro-ecological zone"))

# --------------------------------------------------------------------------------------
# 2. DROUGHT PREDICTION
# --------------------------------------------------------------------------------------
elif page == "Drought Prediction":
    hero("Drought Risk Prediction", "Choose a site, adjust the climate conditions and get an instant risk assessment.")
    rng = meta["feature_ranges"]
    inp, out = st.columns([1, 1.35], gap="large")

    with inp:
        st.markdown("#### 1 · Location")
        loc = st.selectbox("Monitoring site", sorted(locs.Location), index=sorted(locs.Location).index("Mekelle")
                           if "Mekelle" in set(locs.Location) else 0, label_visibility="collapsed")
        row = locs[locs.Location == loc].iloc[0]
        zone = row.Zone
        st.markdown(f'<span class="chip">{row.Region}</span><span class="chip">{zone} zone</span>',
                    unsafe_allow_html=True)
        ldf = df[df.Location == loc]
        typical = {f: float(ldf[f].median()) for f in tm.RAW_FEATURES}

        st.markdown("#### 2 · Climate conditions")
        st.caption("Values start at the site's typical (median) conditions. Move the sliders to test a scenario.")
        vals = {}
        specs = [("Rainfall_mm", 1.0, "%d"), ("Max_Temp_C", 0.1, "%.1f"), ("Min_Temp_C", 0.1, "%.1f"),
                 ("Humidity_pct", 0.1, "%.1f"), ("Evapotranspiration_mm", 0.01, "%.2f"),
                 ("Soil_Moisture_pct", 0.1, "%.1f")]
        for f, step, fmt in specs:
            lo, hi = rng[f]["min"], rng[f]["max"]
            default = min(max(typical[f], lo), hi)
            if f == "Rainfall_mm":
                vals[f] = st.slider(LABELS[f], int(lo), int(hi), int(round(default)), int(step), key=f"{f}_{loc}")
            else:
                vals[f] = st.slider(LABELS[f], float(lo), float(hi), float(round(default, 2)), step, format=fmt,
                                    key=f"{f}_{loc}")

    probs = predict_proba(vals)
    risk = probs.idxmax()
    idx = risk_index(probs)

    with out:
        st.markdown(
            f'<div class="result" style="background:{RISK_SOFT[risk]}">'
            f'<div class="tag">Predicted drought risk · {loc}</div>'
            f'<div class="big" style="color:{RISK_COLORS[risk]}">{risk}</div>'
            f'<div class="small">Confidence {pct(probs[risk], 0)} · Drought risk index <b>{idx:.0f}/100</b></div></div>',
            unsafe_allow_html=True)

        fig = go.Figure(go.Bar(y=CLASSES[::-1], x=[probs[k] * 100 for k in CLASSES[::-1]], orientation="h",
                               marker_color=[RISK_COLORS[k] for k in CLASSES[::-1]],
                               text=[f"{probs[k] * 100:.0f}%" for k in CLASSES[::-1]], textposition="outside"))
        fig.update_layout(xaxis=dict(range=[0, 115], title="Probability (%)"))
        show(style(fig, 210, legend=False).update_layout(title="Class probabilities", margin=dict(l=10, r=10, t=40, b=10)))

        crops, basis, n = recommend_crops(risk, zone)
        st.markdown("##### 🌱 Recommended crops")
        chips = "".join(f'<span class="chip{" main" if i == 0 else ""}">{c} · {pct(s, 0)}</span>'
                        for i, (c, s) in enumerate(crops.items()))
        st.markdown(chips, unsafe_allow_html=True)
        st.caption(f"Share of {n} similar records ({risk.lower()}, {basis}) where the crop was recommended.")

        st.markdown("##### ✅ Recommended actions")
        st.markdown('<ul class="action">' + "".join(f"<li>{a}</li>" for a in action_list(risk)) + "</ul>",
                    unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        lo, hi = rng["Rainfall_mm"]["min"], min(rng["Rainfall_mm"]["max"], 1400)
        grid = np.linspace(lo, hi, 70)
        curve = pd.DataFrame([predict_proba({**vals, "Rainfall_mm": r}) for r in grid])
        fig = go.Figure()
        for k in CLASSES:
            fig.add_trace(go.Scatter(x=grid, y=curve[k] * 100, name=k, stackgroup="one", mode="lines",
                                     line=dict(width=0.5, color=RISK_COLORS[k]), fillcolor=RISK_COLORS[k]))
        fig.add_vline(x=vals["Rainfall_mm"], line_dash="dash", line_color="#12331f",
                      annotation_text="your scenario", annotation_position="top")
        fig.update_layout(xaxis_title="Rainfall (mm)", yaxis_title="Probability (%)")
        show(style(fig, 330).update_layout(title="What-if: how rainfall changes the risk (other inputs fixed)"))
    with c2:
        cmp_df = pd.DataFrame({
            "Indicator": [LABELS[f] for f in tm.RAW_FEATURES],
            "Scenario": [vals[f] for f in tm.RAW_FEATURES],
            "Typical": [typical[f] for f in tm.RAW_FEATURES],
        })
        cmp_df["Change"] = cmp_df["Scenario"] - cmp_df["Typical"]
        st.markdown("**Scenario vs typical for this site**")
        st.dataframe(cmp_df.style.format({"Scenario": "{:.1f}", "Typical": "{:.1f}", "Change": "{:+.1f}"}),
                     hide_index=True, width="stretch")
        rb = meta.get("rainfall_by_class", {})
        if rb:
            callout("<b>Reading the result:</b> in this dataset the risk labels follow rainfall bands – severe up to "
                    f"{rb['Severe Drought']['max']:.0f} mm, moderate {rb['Moderate Drought']['min']:.0f}–"
                    f"{rb['Moderate Drought']['max']:.0f} mm, no drought from {rb['No Drought']['min']:.0f} mm.", warn=True)

# --------------------------------------------------------------------------------------
# 3. REGIONAL ANALYSIS
# --------------------------------------------------------------------------------------
elif page == "Regional Analysis":
    hero("Regional Analysis", "Compare drought risk and climate drivers across regions, zones and sites.")
    f1, f2 = st.columns(2)
    regions = f1.multiselect("Regions", sorted(df.Region.unique()), default=sorted(df.Region.unique()))
    zones = f2.multiselect("Agro-ecological zones", sorted(df.Agro_Ecological_Zone.unique()),
                           default=sorted(df.Agro_Ecological_Zone.unique()))
    d = df[df.Region.isin(regions) & df.Agro_Ecological_Zone.isin(zones)]
    if d.empty:
        st.warning("No records match these filters.")
        st.stop()

    c = st.columns(4)
    c[0].markdown(kpi("Records", f"{len(d):,}", f"{d.Location.nunique()} sites"), unsafe_allow_html=True)
    c[1].markdown(kpi("Drought share", pct(d.Is_Drought.mean()), "moderate + severe"), unsafe_allow_html=True)
    c[2].markdown(kpi("Median rainfall", f"{d.Rainfall_mm.median():.0f} mm", ""), unsafe_allow_html=True)
    c[3].markdown(kpi("Median soil moisture", f"{d.Soil_Moisture_pct.median():.1f}%", ""), unsafe_allow_html=True)
    st.write("")

    t1, t2, t3 = st.tabs(["Risk by place", "Climate drivers", "Data table"])
    with t1:
        s = d.groupby(["Location", "Drought_Risk_Category"], observed=True).size().unstack(fill_value=0)
        s = s.reindex(columns=CLASSES, fill_value=0)
        s = s.div(s.sum(1), axis=0) * 100
        s = s.sort_values(["Severe Drought", "Moderate Drought"], ascending=True)
        s = s[(s["Moderate Drought"] + s["Severe Drought"]) > 0] if st.toggle("Only sites with drought records", True) else s
        fig = go.Figure([go.Bar(y=s.index, x=s[k], name=k, orientation="h", marker_color=RISK_COLORS[k])
                         for k in CLASSES])
        fig.update_layout(barmode="stack", xaxis_title="% of records")
        show(style(fig, max(320, 22 * len(s) + 90)).update_layout(title="Risk mix by site"))
    with t2:
        a, b = st.columns(2)
        with a:
            fig = px.box(d, x="Drought_Risk_Category", y="Rainfall_mm", color="Drought_Risk_Category",
                         color_discrete_map=RISK_COLORS, category_orders={"Drought_Risk_Category": CLASSES})
            fig.update_layout(xaxis_title=None, yaxis_title="Rainfall (mm)")
            show(style(fig, 360, legend=False).update_layout(title="Rainfall by risk category"))
        with b:
            fig = px.scatter(d, x="Rainfall_mm", y="Soil_Moisture_pct", color="Drought_Risk_Category",
                             color_discrete_map=RISK_COLORS, category_orders={"Drought_Risk_Category": CLASSES},
                             opacity=0.7, hover_data=["Location"])
            fig.update_layout(xaxis_title="Rainfall (mm)", yaxis_title="Soil moisture (%)")
            show(style(fig, 360).update_layout(title="Rainfall vs soil moisture"))
        a, b = st.columns(2)
        with a:
            fig = px.scatter(d, x="Max_Temp_C", y="Humidity_pct", color="Drought_Risk_Category",
                             color_discrete_map=RISK_COLORS, category_orders={"Drought_Risk_Category": CLASSES},
                             opacity=0.7, hover_data=["Location"])
            fig.update_layout(xaxis_title="Max temperature (°C)", yaxis_title="Humidity (%)")
            show(style(fig, 360).update_layout(title="Heat vs humidity"))
        with b:
            corr = d[tm.RAW_FEATURES + ["Is_Drought"]].rename(columns={**LABELS, "Is_Drought": "Drought (0/1)"}).corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn_r", zmin=-1, zmax=1, aspect="auto")
            fig.update_xaxes(tickangle=-35)
            show(style(fig, 360, legend=False).update_layout(title="Correlation between variables",
                                                             coloraxis_showscale=False))
    with t3:
        tbl = locs[locs.Location.isin(d.Location)].sort_values("Drought_Share", ascending=False)[
            ["Location", "Region", "Zone", "Observations", "Drought_Share", "Severe_Share", "Rainfall", "MaxTemp", "SoilMoisture"]]
        tbl = tbl.rename(columns={"Drought_Share": "Drought share", "Severe_Share": "Severe share",
                                  "Rainfall": "Median rainfall (mm)", "MaxTemp": "Median max temp (°C)",
                                  "SoilMoisture": "Median soil moisture (%)"})
        st.dataframe(tbl.style.format({"Drought share": "{:.1%}", "Severe share": "{:.1%}",
                                       "Median rainfall (mm)": "{:.0f}", "Median max temp (°C)": "{:.1f}",
                                       "Median soil moisture (%)": "{:.1f}"})
                     .background_gradient(subset=["Drought share"], cmap="YlOrRd"),
                     hide_index=True, width="stretch", height=430)
        st.download_button("Download site summary (CSV)", tbl.to_csv(index=False).encode("utf-8"),
                           "drought_site_summary.csv", "text/csv")

# --------------------------------------------------------------------------------------
# 4. MODEL INSIGHTS
# --------------------------------------------------------------------------------------
elif page == "Model Insights":
    hero("Model Insights", "How the model was built, how well it performs, and what drives its predictions.")
    h = meta["holdout"]
    best = next((r for r in meta["comparison"] if r["model"] == meta["model_name"]), meta["comparison"][0])
    c = st.columns(4)
    c[0].markdown(kpi("Model", meta["model_name"], f"trained {meta['trained_at']}"), unsafe_allow_html=True)
    c[1].markdown(kpi("Test accuracy", pct(h["accuracy"]), f"{h['n_test']} unseen records"), unsafe_allow_html=True)
    c[2].markdown(kpi("Macro F1 (test)", f"{h['macro_f1']:.3f}", "balances all 3 classes"), unsafe_allow_html=True)
    c[3].markdown(kpi("Site-held-out F1", f"{best['location_cv_macro_f1']:.3f}", "tested on unseen sites"),
                  unsafe_allow_html=True)
    st.write("")

    rb, st_t = meta.get("rainfall_by_class", {}), meta.get("stress_test", {})
    if rb:
        callout(
            "<b>Read the accuracy with care.</b> In this dataset the drought label follows rainfall bands almost "
            f"exactly (severe ≤ {rb['Severe Drought']['max']:.0f} mm, moderate {rb['Moderate Drought']['min']:.0f}–"
            f"{rb['Moderate Drought']['max']:.0f} mm, none ≥ {rb['No Drought']['min']:.0f} mm). A near-perfect score "
            "therefore shows the model learned that rule, not that it forecasts real drought. "
            f"Without any rainfall input the macro F1 falls to {st_t.get('macro_f1_without_rainfall', float('nan')):.2f}. "
            "Validate against independent data (e.g. observed crop loss or official SPI) before operational use.",
            warn=True)

    a, b = st.columns(2)
    with a:
        comp = pd.DataFrame(meta["comparison"])
        fig = go.Figure()
        fig.add_bar(x=comp["model"], y=comp["macro_f1"], name="Random 5-fold CV", marker_color=GREEN)
        fig.add_bar(x=comp["model"], y=comp["location_cv_macro_f1"], name="Unseen-site CV", marker_color="#8fc9a5")
        fig.update_layout(barmode="group", yaxis=dict(range=[min(0.8, comp[["macro_f1", "location_cv_macro_f1"]].min().min() - 0.05), 1.01],
                                                      title="Macro F1"))
        show(style(fig, 380).update_layout(title="Model comparison"))
        if meta.get("quick_mode"):
            st.caption("Quick mode: only one model was compared. Run `python train_model.py` to compare all models.")
    with b:
        cm = np.array(h["confusion_matrix"])
        fig = go.Figure(go.Heatmap(z=cm, x=[f"Pred: {k.split()[0]}" for k in CLASSES],
                                   y=[f"True: {k.split()[0]}" for k in CLASSES],
                                   text=cm, texttemplate="%{text}", colorscale=[[0, "#f4f8f5"], [1, GREEN]],
                                   showscale=False, hovertemplate="%{y} → %{x}: %{z}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 380, legend=False).update_layout(title=f"Confusion matrix ({h['n_test']} test records)"))

    a, b = st.columns(2)
    with a:
        imp = pd.DataFrame(meta["importance"]).sort_values("importance")
        imp["name"] = imp["feature"].map(LABELS)
        fig = go.Figure(go.Bar(y=imp["name"], x=imp["importance"], orientation="h", marker_color=GREEN,
                               error_x=dict(type="data", array=imp["std"], color="#7a8c80")))
        fig.update_layout(xaxis_title="Drop in macro F1 when the feature is shuffled")
        show(style(fig, 380, legend=False).update_layout(title="Feature importance (permutation)"))
    with b:
        pc = pd.DataFrame(h["per_class"]).T.loc[CLASSES]
        pc.columns = ["Precision", "Recall", "F1", "Test records"]
        st.markdown("**Per-class results (hold-out)**")
        st.dataframe(pc.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}", "Test records": "{:.0f}"}),
                     width="stretch")
        st.markdown("**Inputs used by the model**")
        st.markdown('<span class="chip">' + '</span><span class="chip">'.join(LABELS[f] for f in meta["features"]) + "</span>",
                    unsafe_allow_html=True)
        st.markdown("**Left out on purpose**")
        for k, v in meta["excluded"].items():
            st.markdown(f"- `{k}` – {v}")

# --------------------------------------------------------------------------------------
# 5. ABOUT
# --------------------------------------------------------------------------------------
else:
    hero("About this system", "Purpose, data, method and limitations.")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("### Purpose")
        st.write("Drought Intelligence classifies agricultural drought risk (No, Moderate or Severe) from climate "
                 "and water-balance indicators and suggests suitable crops and field actions for farmers, extension "
                 "workers and planners in Ethiopia.")
        st.markdown("### Data")
        st.write(f"{meta['n_rows']:,} records from {df.Location.nunique()} sites across {df.Region.nunique()} "
                 "regions and 4 agro-ecological zones (Bereha, Kolla, Woina Dega, Dega). Inputs: rainfall, maximum and "
                 "minimum temperature, humidity, evapotranspiration and soil moisture.")
        st.markdown("### Method")
        st.markdown(
            "- Three engineered features: mean temperature, temperature range and rainfall / evapotranspiration.\n"
            "- Four classifiers compared with stratified and unseen-site cross-validation; best macro F1 selected.\n"
            "- Class weights handle the small number of drought records.\n"
            "- Crop and action advice comes from the dataset's recommendations for the same risk level and zone.")
    with b:
        st.markdown("### Limitations")
        callout("Soil moisture and evapotranspiration in the source file are <b>estimated proxies</b>, there are no "
                "dates, and drought labels follow rainfall bands. The model cannot forecast future drought and is a "
                "decision-support prototype, not an official early-warning product.", warn=True)
        st.markdown(
            "- Replace proxies with measured or satellite soil moisture and station evapotranspiration.\n"
            "- Add dates to enable seasonal and time-series forecasting.\n"
            "- Use official SPI/SPEI from long rainfall records for the drought label.\n"
            "- Confirm recommendations with local agricultural experts.")
        st.markdown("### Retraining")
        st.code("python train_model.py --data data/Ethiopia_Drought_Risk_Dataset.xlsx", language="bash")
        st.caption("Then commit the files in `models/` and `data/` and redeploy.")
