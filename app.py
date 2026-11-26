"""Ethiopia Drought Risk Intelligence Dashboard (Streamlit).

Loads the model trained in Colab (models/*.pkl), the cleaned data (data/dashboard_data.csv)
and the training summary (models/metadata.json). It never retrains anything, so it starts fast.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Ethiopia Drought Risk Intelligence", page_icon="🌾",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------- settings
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_FILE = BASE_DIR / "data" / "dashboard_data.csv"
META_FILE = MODEL_DIR / "metadata.json"

CLASS_NAMES = ["No Drought", "Moderate Drought", "Severe Drought"]
RISK_COLORS = {"No Drought": "#2e8b57", "Moderate Drought": "#f0a030", "Severe Drought": "#c0392b"}
RAW_FEATURES = ["Rainfall_mm", "Max_Temp_C", "Min_Temp_C", "Humidity_pct",
                "Evapotranspiration_mm", "Soil_Moisture_pct"]
DEFAULT_FEATURES = RAW_FEATURES + ["Mean_Temp_C", "Temp_Range_C", "Rain_ET_Ratio"]
LABELS = {"Rainfall_mm": "Rainfall (mm)", "Max_Temp_C": "Max temperature (°C)",
          "Min_Temp_C": "Min temperature (°C)", "Humidity_pct": "Humidity (%)",
          "Evapotranspiration_mm": "Evapotranspiration (mm)", "Soil_Moisture_pct": "Soil moisture (%)"}
SEVERE_MAX_MM, MODERATE_MAX_MM = 228, 416

PRESETS = {
    "Very dry season (Afar-like)": dict(Rainfall_mm=150, Max_Temp_C=38.0, Min_Temp_C=24.0, Humidity_pct=30,
                                        Evapotranspiration_mm=8.8, Soil_Moisture_pct=10),
    "Below-normal rainfall": dict(Rainfall_mm=320, Max_Temp_C=33.0, Min_Temp_C=20.0, Humidity_pct=42,
                                  Evapotranspiration_mm=7.8, Soil_Moisture_pct=18),
    "Good rainfall (highlands)": dict(Rainfall_mm=1000, Max_Temp_C=22.0, Min_Temp_C=10.0, Humidity_pct=68,
                                      Evapotranspiration_mm=5.5, Soil_Moisture_pct=42),
}
SLIDER_BOUNDS = {"Rainfall_mm": (0, 2500, 10), "Max_Temp_C": (5.0, 50.0, 0.5),
                 "Min_Temp_C": (-5.0, 35.0, 0.5), "Humidity_pct": (0, 100, 1),
                 "Evapotranspiration_mm": (0.5, 15.0, 0.1), "Soil_Moisture_pct": (0, 100, 1)}

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg,#1f6b45,#2e8b57 60%,#58a97a); padding: 1.3rem 1.6rem;
       border-radius: 14px; color: #fff; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 1.85rem; color: #fff;}
.hero p {margin: .3rem 0 0; opacity: .93;}
.risk-card {border-radius: 14px; padding: 1.1rem 1.3rem; color: #fff; margin-bottom: .6rem;}
.risk-card .small {font-size: .85rem; opacity: .9; text-transform: uppercase; letter-spacing: .05em;}
.risk-card .big {font-size: 1.9rem; font-weight: 700; line-height: 1.2;}
.note {border-left: 4px solid #f0a030; padding: .6rem .9rem; background: rgba(240,160,48,.10);
       border-radius: 6px; font-size: .92rem;}
div[data-testid="stMetric"] {background: rgba(46,139,87,.08); border: 1px solid rgba(46,139,87,.25);
       padding: .7rem .9rem; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    files = sorted(MODEL_DIR.glob("*.pkl"))
    return joblib.load(files[0]) if files else None


@st.cache_data
def load_meta():
    return json.loads(META_FILE.read_text()) if META_FILE.exists() else {}


@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        return None
    d = pd.read_csv(DATA_FILE)
    for c in ["Region", "Location", "Agro_Ecological_Zone", "Drought_Risk"]:
        d[c] = d[c].astype(str).str.strip()
    return d


model, meta, data = load_model(), load_meta(), load_data()
if model is None or data is None:
    st.error("Model or data file not found. Upload `models/*.pkl` and `data/dashboard_data.csv` "
             "to the same GitHub folder as `app.py`.")
    st.write("Files found next to app.py:", sorted(p.name for p in BASE_DIR.rglob("*") if p.is_file())[:50])
    st.stop()

FEATURES = meta.get("features", DEFAULT_FEATURES)
ZONES = sorted(data["Agro_Ecological_Zone"].dropna().unique())
MODEL_NAME = meta.get("model_name", type(model.named_steps["clf"]).__name__ if hasattr(model, "named_steps") else "Model")


# ----------------------------------------------------------------------------- model helpers
def add_engineered(frame):
    out = frame.copy()
    out["Mean_Temp_C"] = (out["Max_Temp_C"] + out["Min_Temp_C"]) / 2
    out["Temp_Range_C"] = out["Max_Temp_C"] - out["Min_Temp_C"]
    out["Rain_ET_Ratio"] = out["Rainfall_mm"] / out["Evapotranspiration_mm"].clip(lower=0.1)
    return out


def predict_proba(frame):
    frame = frame.copy()
    if "Agro_Ecological_Zone" not in frame:
        frame["Agro_Ecological_Zone"] = None
    return model.predict_proba(add_engineered(frame)[FEATURES])


def risk_index(proba):
    """0-100: 0 = no drought, 100 = certain severe drought."""
    return 100 * (0.5 * proba[:, 1] + proba[:, 2])


def recommend_crops(risk, zone, top=5):
    subset = data[(data["Drought_Risk"] == risk) & (data["Agro_Ecological_Zone"] == zone)]
    if len(subset) < 8:
        subset = data[data["Drought_Risk"] == risk]
    if subset.empty:
        return []
    crops = subset["Recommended_Crops"].astype(str).str.split(",").explode().str.strip()
    share = (crops.value_counts() / len(subset)).head(top)
    return [(c, v) for c, v in share.items()]


def recommend_actions(risk):
    sub = data.loc[data["Drought_Risk"] == risk, "Recommended_Actions"].dropna()
    return sub.mode().iat[0] if not sub.empty else "No recommendation available."


def gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": " / 100"},
        title={"text": "Drought risk index"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#222"},
               "steps": [{"range": [0, 33], "color": RISK_COLORS["No Drought"]},
                         {"range": [33, 66], "color": RISK_COLORS["Moderate Drought"]},
                         {"range": [66, 100], "color": RISK_COLORS["Severe Drought"]}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# ----------------------------------------------------------------------------- header + sidebar
st.markdown(f"""
<div class="hero"><h1>🌾 Ethiopia Drought Risk Intelligence</h1>
<p>Predict drought risk from climate conditions and get crop and field-action recommendations.
Model: <b>{MODEL_NAME}</b> · trained on {meta.get('n_rows', len(data)):,} records · {data['Location'].nunique()} sites</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("About")
    st.write("This dashboard turns a machine-learning model into a decision-support tool for "
             "farmers, extension workers and planners.")
    st.metric("Sites covered", data["Location"].nunique())
    st.metric("Regions", data["Region"].nunique())
    if meta.get("trained_at"):
        st.caption(f"Model trained: {meta['trained_at']}")
    st.caption("Risk index = 100 × (0.5 × P(moderate) + 1.0 × P(severe)).")

tab_pred, tab_batch, tab_region, tab_insight, tab_model = st.tabs(
    ["🎯 Risk Predictor", "📁 Batch Prediction", "🗺️ Regional Explorer", "📊 Data Insights", "🧪 Model Performance"])

# ----------------------------------------------------------------------------- tab 1: predictor
with tab_pred:
    for k, v in PRESETS["Below-normal rainfall"].items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("zone", ZONES[0])

    def apply_preset():
        for k, v in PRESETS[st.session_state["preset"]].items():
            st.session_state[k] = v

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("Enter conditions")
        st.selectbox("Quick scenario", list(PRESETS), index=1, key="preset", on_change=apply_preset)
        st.selectbox("Agro-ecological zone (used for crop advice)", ZONES, key="zone")
        c1, c2 = st.columns(2)
        values = {}
        for i, f in enumerate(RAW_FEATURES):
            lo, hi, step = SLIDER_BOUNDS[f]
            with (c1 if i % 2 == 0 else c2):
                values[f] = st.slider(LABELS[f], lo, hi, key=f, step=step)

    row = pd.DataFrame([{**values, "Agro_Ecological_Zone": st.session_state["zone"]}])
    with right:
        st.subheader("Prediction")
        if values["Min_Temp_C"] > values["Max_Temp_C"]:
            st.error("Minimum temperature cannot be higher than maximum temperature.")
        else:
            proba = predict_proba(row)
            idx = float(risk_index(proba)[0])
            pred = CLASS_NAMES[int(proba.argmax())]
            st.markdown(f'<div class="risk-card" style="background:{RISK_COLORS[pred]}">'
                        f'<div class="small">Predicted drought risk</div><div class="big">{pred}</div>'
                        f'<div>Confidence {proba.max():.0%}</div></div>', unsafe_allow_html=True)
            g1, g2 = st.columns([1, 1])
            g1.plotly_chart(gauge(idx), width="stretch")
            pfig = px.bar(x=CLASS_NAMES, y=proba[0] * 100, color=CLASS_NAMES, color_discrete_map=RISK_COLORS,
                          labels={"x": "", "y": "Probability (%)"}, title="Class probabilities")
            pfig.update_layout(height=260, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
            g2.plotly_chart(pfig, width="stretch")

            outside = [LABELS[f] for f in RAW_FEATURES
                       if not data[f].min() <= values[f] <= data[f].max()]
            if outside:
                st.warning("Outside the range the model was trained on: " + ", ".join(outside) +
                           ". Treat this prediction with extra caution.")

    if values["Min_Temp_C"] <= values["Max_Temp_C"]:
        st.divider()
        r1, r2 = st.columns([1, 1], gap="large")
        with r1:
            st.subheader("Recommended crops")
            crops = recommend_crops(pred, st.session_state["zone"])
            if crops:
                cdf = pd.DataFrame(crops, columns=["Crop", "Share"])
                cfig = px.bar(cdf.iloc[::-1], x="Share", y="Crop", orientation="h", text=cdf.iloc[::-1]["Share"].map("{:.0%}".format))
                cfig.update_traces(marker_color=RISK_COLORS[pred])
                cfig.update_layout(height=280, xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(cfig, width="stretch")
                st.caption("Share of similar records in the dataset that recommend each crop.")
            else:
                st.info("No crop recommendations available.")
        with r2:
            st.subheader("Recommended field actions")
            st.success(recommend_actions(pred))
            st.subheader("Your inputs vs typical values")
            typical = data[data["Drought_Risk"] == pred][RAW_FEATURES].median()
            cmp_df = pd.DataFrame({"Your input": pd.Series(values), f"Typical for '{pred}'": typical}).round(1)
            cmp_df.index = [LABELS[i] for i in cmp_df.index]
            st.dataframe(cmp_df, width="stretch")

        st.subheader("What if rainfall changes?")
        rain = np.linspace(0, 1400, 57)
        sweep = pd.DataFrame([{**values, "Agro_Ecological_Zone": st.session_state["zone"]}] * len(rain))
        sweep["Rainfall_mm"] = rain
        curve = risk_index(predict_proba(sweep))
        wf = go.Figure(go.Scatter(x=rain, y=curve, mode="lines", line=dict(width=3, color="#1f6b45"), name="Risk index"))
        wf.add_trace(go.Scatter(x=[values["Rainfall_mm"]], y=[idx], mode="markers", name="Current input",
                                marker=dict(size=14, color=RISK_COLORS[pred], line=dict(width=2, color="white"))))
        wf.add_vrect(x0=0, x1=SEVERE_MAX_MM, fillcolor=RISK_COLORS["Severe Drought"], opacity=0.08, line_width=0)
        wf.add_vrect(x0=SEVERE_MAX_MM, x1=MODERATE_MAX_MM, fillcolor=RISK_COLORS["Moderate Drought"], opacity=0.10, line_width=0)
        wf.update_layout(height=320, xaxis_title="Rainfall (mm), other inputs held constant",
                         yaxis_title="Risk index (0-100)", margin=dict(l=10, r=10, t=10, b=10),
                         legend=dict(orientation="h", y=1.1))
        st.plotly_chart(wf, width="stretch")

# ----------------------------------------------------------------------------- tab 2: batch
with tab_batch:
    st.subheader("Predict many locations at once")
    st.write("Upload a CSV with these columns: " + ", ".join(f"`{c}`" for c in RAW_FEATURES) +
             ". Optional: `Region`, `Location`, `Agro_Ecological_Zone`.")
    template = pd.DataFrame([PRESETS[k] for k in PRESETS]).assign(Location=["Site A", "Site B", "Site C"])
    st.download_button("Download CSV template", template.to_csv(index=False).encode(), "template.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        try:
            batch = pd.read_csv(up)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            st.stop()
        missing = [c for c in RAW_FEATURES if c not in batch.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            for c in RAW_FEATURES:
                batch[c] = pd.to_numeric(batch[c], errors="coerce")
            ok = batch[RAW_FEATURES].notna().all(axis=1) & (batch["Min_Temp_C"] <= batch["Max_Temp_C"])
            if (~ok).any():
                st.warning(f"{(~ok).sum()} row(s) skipped: missing values or min temperature above max.")
            good = batch[ok].copy()
            if good.empty:
                st.error("No valid rows to predict.")
            else:
                pb = predict_proba(good)
                good["Predicted_Risk"] = [CLASS_NAMES[i] for i in pb.argmax(axis=1)]
                good["Confidence"] = pb.max(axis=1).round(3)
                good["Risk_Index"] = risk_index(pb).round(1)
                zone_col = good["Agro_Ecological_Zone"] if "Agro_Ecological_Zone" in good else [None] * len(good)
                good["Top_Crops"] = [", ".join(c for c, _ in recommend_crops(r, z, 3))
                                     for r, z in zip(good["Predicted_Risk"], zone_col)]
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows predicted", len(good))
                m2.metric("In drought", f"{(good['Predicted_Risk'] != 'No Drought').mean():.0%}")
                m3.metric("Average risk index", f"{good['Risk_Index'].mean():.1f}")
                bf = px.histogram(good, x="Predicted_Risk", color="Predicted_Risk", color_discrete_map=RISK_COLORS,
                                  category_orders={"Predicted_Risk": CLASS_NAMES})
                bf.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(bf, width="stretch")
                st.dataframe(good, width="stretch")
                st.download_button("Download predictions", good.to_csv(index=False).encode(),
                                   "drought_predictions.csv", "text/csv")

# ----------------------------------------------------------------------------- tab 3: regional explorer
with tab_region:
    st.subheader("Explore drought patterns in the data")
    f1, f2 = st.columns(2)
    regions = f1.multiselect("Region", sorted(data["Region"].unique()), default=sorted(data["Region"].unique()))
    zones = f2.multiselect("Agro-ecological zone", ZONES, default=ZONES)
    sub = data[data["Region"].isin(regions) & data["Agro_Ecological_Zone"].isin(zones)]
    if sub.empty:
        st.info("No records match these filters.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Records", f"{len(sub):,}")
        k2.metric("In drought", f"{(sub['Drought_Risk'] != 'No Drought').mean():.0%}")
        k3.metric("Severe drought", f"{(sub['Drought_Risk'] == 'Severe Drought').mean():.0%}")
        k4.metric("Average rainfall", f"{sub['Rainfall_mm'].mean():.0f} mm")

        c1, c2 = st.columns(2)
        mix = (sub.groupby(["Region", "Drought_Risk"]).size().rename("n").reset_index())
        mix["pct"] = mix["n"] / mix.groupby("Region")["n"].transform("sum") * 100
        fig = px.bar(mix, x="pct", y="Region", color="Drought_Risk", orientation="h", color_discrete_map=RISK_COLORS,
                     category_orders={"Drought_Risk": CLASS_NAMES}, labels={"pct": "% of records"},
                     title="Risk mix by region")
        fig.update_layout(height=380, legend_title_text="")
        c1.plotly_chart(fig, width="stretch")
        fig = px.scatter(sub, x="Rainfall_mm", y="Soil_Moisture_pct", color="Drought_Risk",
                         color_discrete_map=RISK_COLORS, category_orders={"Drought_Risk": CLASS_NAMES},
                         hover_data=["Location", "Region"], opacity=0.7, title="Rainfall vs soil moisture")
        fig.update_layout(height=380, legend_title_text="")
        c2.plotly_chart(fig, width="stretch")

        st.markdown("**Most drought-prone sites** (sites with at least 5 records)")
        sites = sub.groupby(["Location", "Region"]).agg(
            Records=("Drought_Risk", "size"),
            Drought_share=("Drought_Risk", lambda s: (s != "No Drought").mean()),
            Severe_share=("Drought_Risk", lambda s: (s == "Severe Drought").mean()),
            Avg_rainfall_mm=("Rainfall_mm", "mean")).reset_index()
        sites = sites[sites["Records"] >= 5].sort_values(["Severe_share", "Drought_share"], ascending=False)
        st.dataframe(sites.style.format({"Drought_share": "{:.0%}", "Severe_share": "{:.0%}", "Avg_rainfall_mm": "{:.0f}"}),
                     width="stretch", hide_index=True)

# ----------------------------------------------------------------------------- tab 4: data insights
with tab_insight:
    st.subheader("How the climate variables relate to drought")
    c1, c2 = st.columns(2)
    counts = data["Drought_Risk"].value_counts().reindex(CLASS_NAMES).fillna(0)
    fig = px.pie(values=counts.values, names=counts.index, hole=0.55, color=counts.index,
                 color_discrete_map=RISK_COLORS, title="Risk category distribution")
    fig.update_layout(height=340)
    c1.plotly_chart(fig, width="stretch")
    feat = c2.selectbox("Variable to compare across risk categories", RAW_FEATURES, format_func=LABELS.get)
    fig = px.box(data, x="Drought_Risk", y=feat, color="Drought_Risk", color_discrete_map=RISK_COLORS,
                 category_orders={"Drought_Risk": CLASS_NAMES}, labels={feat: LABELS[feat], "Drought_Risk": ""})
    fig.update_layout(height=340, showlegend=False, margin=dict(t=30))
    c2.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    corr = data[RAW_FEATURES].assign(Is_Drought=(data["Drought_Risk"] != "No Drought").astype(int)).corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdYlGn_r", zmin=-1, zmax=1,
                    title="Correlation matrix")
    fig.update_layout(height=420)
    c3.plotly_chart(fig, width="stretch")
    zone_mix = (data.groupby(["Agro_Ecological_Zone", "Drought_Risk"]).size().rename("n").reset_index())
    zone_mix["pct"] = zone_mix["n"] / zone_mix.groupby("Agro_Ecological_Zone")["n"].transform("sum") * 100
    fig = px.bar(zone_mix, x="Agro_Ecological_Zone", y="pct", color="Drought_Risk", color_discrete_map=RISK_COLORS,
                 category_orders={"Drought_Risk": CLASS_NAMES}, labels={"pct": "% of records", "Agro_Ecological_Zone": ""},
                 title="Risk mix by agro-ecological zone")
    fig.update_layout(height=420, legend_title_text="")
    c4.plotly_chart(fig, width="stretch")

    with st.expander("Browse the data"):
        st.dataframe(data, width="stretch", hide_index=True)
        st.download_button("Download data (CSV)", data.to_csv(index=False).encode(), "drought_data.csv", "text/csv")

# ----------------------------------------------------------------------------- tab 5: model performance
with tab_model:
    st.subheader("How reliable is the model?")
    hold = meta.get("holdout")
    if not hold:
        st.info("`models/metadata.json` was not found, so model scores are not available. "
                "Run the export cell in Colab and upload that file.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model", MODEL_NAME)
        m2.metric("Test accuracy", f"{hold['accuracy']:.1%}")
        m3.metric("Test macro F1", f"{hold['macro_f1']:.3f}")
        m4.metric("Balanced accuracy", f"{hold['balanced_accuracy']:.1%}")

        c1, c2 = st.columns(2)
        cm = np.array(hold["confusion_matrix"])
        short = [c.split()[0] for c in CLASS_NAMES]
        fig = px.imshow(cm, x=short, y=short, text_auto=True, color_continuous_scale="Greens",
                        labels=dict(x="Predicted", y="True"), title=f"Confusion matrix (test set, n={hold['n_test']})")
        fig.update_layout(height=380, coloraxis_showscale=False)
        c1.plotly_chart(fig, width="stretch")
        pc = pd.DataFrame(hold["per_class"]).T.rename(columns={"f1-score": "f1"})
        c2.markdown("**Per-class results**")
        c2.dataframe(pc.round(3), width="stretch")

        comp = pd.DataFrame(meta.get("comparison", []))
        if not comp.empty:
            long = comp.melt(id_vars="model", value_vars=["macro_f1", "location_cv_macro_f1"],
                             var_name="Metric", value_name="Macro F1")
            long["Metric"] = long["Metric"].map({"macro_f1": "Cross-validation", "location_cv_macro_f1": "Unseen sites"})
            fig = px.bar(long, x="model", y="Macro F1", color="Metric", barmode="group",
                         color_discrete_sequence=["#1f6b45", "#a8d5b9"], title="Model comparison (macro F1)")
            fig.update_layout(height=340, xaxis_title="")
            st.plotly_chart(fig, width="stretch")

        imp = pd.DataFrame(meta.get("importance", []))
        if not imp.empty:
            imp = imp.sort_values("importance")
            fig = px.bar(imp, x="importance", y="feature", orientation="h", error_x="std",
                         title="Feature importance (drop in macro F1 when shuffled)")
            fig.update_traces(marker_color="#1f6b45")
            fig.update_layout(height=380, yaxis_title="")
            st.plotly_chart(fig, width="stretch")

        stress = meta.get("stress_test")
        if stress:
            st.write(f"Without any rainfall information the model still reaches macro F1 "
                     f"**{stress['macro_f1_without_rainfall']:.3f}** and accuracy "
                     f"**{stress['accuracy_without_rainfall']:.1%}**.")

    rule = np.where(data["Rainfall_mm"] <= SEVERE_MAX_MM, "Severe Drought",
                    np.where(data["Rainfall_mm"] <= MODERATE_MAX_MM, "Moderate Drought", "No Drought"))
    agree = (rule == data["Drought_Risk"]).mean()
    st.markdown(f'<div class="note"><b>Interpret with care:</b> the drought labels in this dataset match a simple '
                f'rainfall rule (≤{SEVERE_MAX_MM} mm severe, ≤{MODERATE_MAX_MM} mm moderate) in <b>{agree:.1%}</b> of records, '
                f'so very high scores are expected. Validate the model against independent data such as official '
                f'SPI/NDVI drought records or observed crop loss before using it for real decisions.</div>',
                unsafe_allow_html=True)

st.caption("Drought Intelligence · Ethiopia · decision-support prototype, not an official forecast.")
