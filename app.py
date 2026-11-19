import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="wide")

ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
COLORS = {
    "Good": "#009865",
    "Satisfactory": "#7BC043",
    "Moderate": "#F2C500",
    "Poor": "#F28C28",
    "Very Poor": "#E03C31",
    "Severe": "#8B0000",
}
ADVICE = {
    "Good": "Air is clean. Enjoy outdoor activities.",
    "Satisfactory": "Fine for most people. Very sensitive people may feel slight discomfort.",
    "Moderate": "People with asthma or heart disease should limit long outdoor activity.",
    "Poor": "Everyone may feel discomfort. Reduce outdoor exercise and wear a mask outside.",
    "Very Poor": "Avoid outdoor activity. Keep windows closed and use an air purifier if you can.",
    "Severe": "Health emergency. Stay indoors and avoid all outdoor exertion.",
}
LABELS = {
    "pm25": "PM2.5 (µg/m³)",
    "pm10": "PM10 (µg/m³)",
    "no2": "NO₂ (µg/m³)",
    "so2": "SO₂ (µg/m³)",
    "co": "CO (mg/m³)",
    "o3": "O₃ (µg/m³)",
    "nh3": "NH₃ (µg/m³)",
    "Temperature_C": "Temperature (°C)",
    "Humidity_%": "Humidity (%)",
    "Wind_Speed_kmh": "Wind speed (km/h)",
}
WEATHER = ["Temperature_C", "Humidity_%", "Wind_Speed_kmh"]


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


@st.cache_data
def load_meta():
    with open("model_meta.json") as f:
        return json.load(f)


model = load_model()
meta = load_meta()
features = meta["features"]
stats = meta["stats"]
pollutants = [f for f in features if f not in WEATHER]


def slider_range(f):
    """Use the 99th percentile as the upper limit when the max is an extreme outlier."""
    s = stats[f]
    lo, hi = float(s["min"]), float(s["max"])
    if hi > 2 * float(s["p99"]) > 0:
        hi = float(s["p99"])
    return lo, hi


def clamp(f, v):
    lo, hi = slider_range(f)
    return min(max(float(v), lo), hi)


def make_preset(pollutant_stat):
    """Pollutants use the chosen statistic; weather stays at its median."""
    return {
        f: clamp(f, stats[f][pollutant_stat] if f in pollutants else stats[f]["median"])
        for f in features
    }


PRESETS = {
    "😊 Clean day": make_preset("p10"),
    "😐 Typical day": make_preset("median"),
    "😷 Polluted day": make_preset("p90"),
    "☠️ Extreme day": make_preset("p99"),
}


def predict_proba_df(df):
    probs = pd.DataFrame(model.predict_proba(df[features]), columns=model.classes_, index=df.index)
    return probs[[c for c in ORDER if c in probs.columns]]


# ---------- Session state ----------
for f in features:
    if f"in_{f}" not in st.session_state:
        st.session_state[f"in_{f}"] = clamp(f, stats[f]["median"])


def apply_preset(name):
    for f, v in PRESETS[name].items():
        st.session_state[f"in_{f}"] = v


# ---------- Header ----------
st.title("🌫️ Air Quality Index (AQI) Predictor")
st.caption("Move the sliders or pick a preset. The prediction updates instantly.")

tab_predict, tab_whatif, tab_batch, tab_about = st.tabs(
    ["🎯 Predict", "🔬 What-if explorer", "📂 Batch (CSV)", "ℹ️ About"]
)

# ================= TAB 1: PREDICT =================
with tab_predict:
    st.write("**Quick presets**")
    pcols = st.columns(len(PRESETS))
    for col, name in zip(pcols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), use_container_width=True)

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Readings")
        grid = st.columns(2)
        for i, f in enumerate(features):
            lo, hi = slider_range(f)
            with grid[i % 2]:
                st.slider(
                    LABELS.get(f, f),
                    min_value=lo,
                    max_value=hi,
                    step=(hi - lo) / 200 if hi > lo else 1.0,
                    key=f"in_{f}",
                )

    current = pd.DataFrame([{f: st.session_state[f"in_{f}"] for f in features}])
    probs = predict_proba_df(current).iloc[0]
    status = probs.idxmax()
    color = COLORS.get(status, "#444")

    with right:
        st.subheader("Prediction")
        st.markdown(
            f"<div style='padding:22px;border-radius:14px;background:{color};color:white;"
            f"text-align:center;font-size:30px;font-weight:700'>{status}</div>",
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{probs.max() * 100:.1f}%")
        st.info(ADVICE.get(status, ""))
        st.write("**Probability of each category**")
        st.bar_chart(probs)

# ================= TAB 2: WHAT-IF =================
with tab_whatif:
    st.subheader("What happens if one value changes?")
    st.write("Pick a reading. Everything else stays as set in the Predict tab.")
    chosen = st.selectbox("Reading to vary", features, format_func=lambda f: LABELS.get(f, f))
    lo, hi = slider_range(chosen)
    sweep_vals = [lo + (hi - lo) * i / 49 for i in range(50)]
    sweep = pd.concat([current] * 50, ignore_index=True)
    sweep[chosen] = sweep_vals
    sweep_probs = predict_proba_df(sweep)
    sweep_probs.index = [round(v, 2) for v in sweep_vals]
    sweep_probs.index.name = LABELS.get(chosen, chosen)
    st.line_chart(sweep_probs)

    sweep_status = sweep_probs.idxmax(axis=1)
    changes = sweep_status[sweep_status != sweep_status.shift()]
    st.write("**Where the predicted category changes:**")
    for value, cat in changes.items():
        st.write(f"- from **{value}** → **{cat}**")

# ================= TAB 3: BATCH =================
with tab_batch:
    st.subheader("Predict many rows at once")
    st.write("Upload a CSV with these columns:")
    st.code(", ".join(features))
    template = pd.DataFrame([PRESETS["😐 Typical day"]] * 3)
    st.download_button(
        "Download sample CSV", template.to_csv(index=False), "sample_input.csv", "text/csv"
    )
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in features if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            data = data.dropna(subset=features).reset_index(drop=True)
            bp = predict_proba_df(data)
            data["Predicted_AQI"] = bp.idxmax(axis=1)
            data["Confidence_%"] = (bp.max(axis=1) * 100).round(1)
            st.success(f"Predicted {len(data)} rows.")
            st.dataframe(data, use_container_width=True)
            st.bar_chart(data["Predicted_AQI"].value_counts().reindex(ORDER).dropna())
            st.download_button(
                "Download results", data.to_csv(index=False), "aqi_predictions.csv", "text/csv"
            )

# ================= TAB 4: ABOUT =================
with tab_about:
    st.subheader("About the model")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{meta['accuracy']:.3f}")
    c2.metric("Macro F1", f"{meta['macro_f1']:.3f}")
    st.write("**Feature importance**")
    st.bar_chart(pd.Series(meta["importance"]).sort_values())
    st.write("**Per-class report**")
    st.dataframe(pd.DataFrame(meta["report"]).T.round(3), use_container_width=True)
