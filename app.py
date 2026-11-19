import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="wide")

CATEGORY_COLORS = {
    "Good": "#009865",
    "Satisfactory": "#7BC043",
    "Moderate": "#F2C500",
    "Poor": "#F28C28",
    "Very Poor": "#E03C31",
    "Severe": "#8B0000",
}
ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

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

st.title("🌫️ Air Quality Index (AQI) Predictor")
st.write("Enter gas and weather readings to predict the AQI category.")

# ---------- Inputs ----------
st.subheader("Enter readings")
values = {}
cols = st.columns(2)
for i, f in enumerate(features):
    s = stats[f]
    with cols[i % 2]:
        values[f] = st.number_input(
            LABELS.get(f, f),
            min_value=float(s["min"]),
            max_value=float(s["max"]),
            value=float(s["median"]),
            help=f"Typical range in the training data: {s['p10']:.1f} to {s['p90']:.1f}",
        )

# ---------- Prediction ----------
if st.button("Predict AQI", type="primary"):
    reading = pd.DataFrame([values], columns=features)
    status = model.predict(reading)[0]
    probs = pd.Series(model.predict_proba(reading)[0], index=model.classes_)
    probs = probs.reindex([c for c in ORDER if c in probs.index])

    color = CATEGORY_COLORS.get(status, "#444")
    st.markdown(
        f"<div style='padding:18px;border-radius:12px;background:{color};"
        f"color:white;text-align:center;font-size:28px;font-weight:700'>"
        f"Predicted AQI category: {status}</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Confidence for each category")
    st.bar_chart(probs)

# ---------- Model info ----------
with st.expander("About the model"):
    st.write("Model: Logistic Regression (with feature scaling)")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{meta['accuracy']:.3f}")
    c2.metric("Macro F1", f"{meta['macro_f1']:.3f}")
    st.write("Feature importance")
    st.bar_chart(pd.Series(meta["importance"]).sort_values())
