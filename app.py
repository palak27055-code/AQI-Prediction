import json
import math
import random

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="wide")

# ---------------------------------------------------------------- constants
ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
COLORS = {
    "Good": "#009865",
    "Satisfactory": "#7BC043",
    "Moderate": "#F2C500",
    "Poor": "#F28C28",
    "Very Poor": "#E03C31",
    "Severe": "#8B0000",
}
MASCOT = {
    "Good": ("😄", "Fresh air! Your lungs are happy.", "ताज़ी हवा! फेफड़े खुश हैं।"),
    "Satisfactory": ("🙂", "Pretty decent air today.", "आज की हवा काफ़ी ठीक है।"),
    "Moderate": ("😐", "Okay, but not great.", "ठीक-ठाक है, पर बढ़िया नहीं।"),
    "Poor": ("😷", "Time to wear a mask!", "मास्क पहनने का समय!"),
    "Very Poor": ("🤢", "Stay inside if you can.", "हो सके तो घर में रहें।"),
    "Severe": ("☠️", "Danger zone! Stay indoors.", "ख़तरा! घर के अंदर रहें।"),
}
ADVICE = {
    "Good": ("Air is clean. Enjoy the outdoors!", "हवा साफ़ है। बाहर घूमने का मज़ा लें।"),
    "Satisfactory": (
        "Fine for most people. Very sensitive people may feel slight discomfort.",
        "ज़्यादातर लोगों के लिए ठीक है। बहुत संवेदनशील लोगों को हल्की परेशानी हो सकती है।",
    ),
    "Moderate": (
        "People with asthma or heart disease should limit long outdoor activity.",
        "अस्थमा या दिल के मरीज़ बाहर ज़्यादा देर न रहें।",
    ),
    "Poor": (
        "Everyone may feel discomfort. Reduce outdoor exercise and wear a mask outside.",
        "सभी को परेशानी हो सकती है। बाहर कसरत कम करें और मास्क पहनें।",
    ),
    "Very Poor": (
        "Avoid outdoor activity. Keep windows closed and use an air purifier if you can.",
        "बाहर जाने से बचें। खिड़कियाँ बंद रखें, हो सके तो एयर प्यूरीफ़ायर चलाएँ।",
    ),
    "Severe": (
        "Health emergency. Stay indoors and avoid all outdoor exertion.",
        "स्वास्थ्य आपातकाल। घर के अंदर रहें और बाहर मेहनत वाला काम न करें।",
    ),
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

# ---------------------------------------------------------------- style
st.markdown(
    """
<style>
.hero{background:linear-gradient(135deg,#0f2027 0%,#2c5364 55%,#e96443 100%);
padding:28px 32px;border-radius:18px;margin-bottom:14px;color:#fff}
.hero h1{margin:0;font-size:2.3rem;color:#fff}
.hero p{margin:6px 0 0 0;font-size:1.05rem;opacity:.92;color:#fff}
.banner{padding:20px;border-radius:16px;color:#fff;text-align:center;
box-shadow:0 6px 18px rgba(0,0,0,.25)}
.banner .big{font-size:56px;line-height:1.1}
.banner .cat{font-size:32px;font-weight:800}
.banner .msg{font-size:16px;opacity:.95}
.stTabs [data-baseweb="tab"]{font-size:1.05rem}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loading
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


# ---------------------------------------------------------------- helpers
def slider_range(f):
    s = stats[f]
    lo, hi = float(s["min"]), float(s["max"])
    if hi > 2 * float(s["p99"]) > 0:
        hi = float(s["p99"])
    return lo, hi


def clamp(f, v):
    lo, hi = slider_range(f)
    return min(max(float(v), lo), hi)


def make_preset(stat):
    return {
        f: clamp(f, stats[f][stat] if f in pollutants else stats[f]["median"])
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


def gauge_svg(probs):
    """Half-circle meter. The needle points to the model's expected pollution level."""
    n = len(probs)
    score = sum(p * i for i, p in enumerate(probs.values)) / max(n - 1, 1)
    cx, cy, r = 150, 150, 115
    seg = 180 / n
    arcs = ""
    for i, cat in enumerate(probs.index):
        a1 = math.radians(180 - i * seg)
        a2 = math.radians(180 - (i + 1) * seg)
        x1, y1 = cx + r * math.cos(a1), cy - r * math.sin(a1)
        x2, y2 = cx + r * math.cos(a2), cy - r * math.sin(a2)
        arcs += (
            f'<path d="M {x1:.1f} {y1:.1f} A {r} {r} 0 0 1 {x2:.1f} {y2:.1f}" '
            f'stroke="{COLORS[cat]}" stroke-width="26" fill="none"/>'
        )
    ang = math.radians(180 - score * 180)
    nx, ny = cx + 98 * math.cos(ang), cy - 98 * math.sin(ang)
    return (
        '<svg viewBox="0 0 300 175" style="width:100%;max-width:420px;display:block;margin:auto">'
        f"{arcs}"
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#888" stroke-width="5" stroke-linecap="round"/>'
        f'<circle cx="{cx}" cy="{cy}" r="9" fill="#888"/>'
        '<text x="22" y="170" font-size="11" fill="#888">Clean</text>'
        '<text x="250" y="170" font-size="11" fill="#888">Toxic</text>'
        "</svg>"
    )


# ---------------------------------------------------------------- state
for f in features:
    if f"in_{f}" not in st.session_state:
        st.session_state[f"in_{f}"] = clamp(f, stats[f]["median"])


def apply_preset(name):
    for f, v in PRESETS[name].items():
        st.session_state[f"in_{f}"] = v


def sample_scenario():
    u = random.random()
    row = {}
    for f in features:
        s = stats[f]
        if f in pollutants:
            t = min(max(u + random.uniform(-0.12, 0.12), 0.0), 1.0)
            v = s["p10"] + (s["p99"] - s["p10"]) * t
        else:
            v = random.uniform(s["p10"], s["p90"])
        row[f] = round(clamp(f, v), 2)
    return row


def new_round():
    st.session_state["g_row"] = sample_scenario()
    st.session_state["g_id"] = st.session_state.get("g_id", 0) + 1
    st.session_state["g_done"] = False
    st.session_state["g_celebrated"] = False


def lock_answer():
    gid = st.session_state["g_id"]
    guess = st.session_state[f"guess_{gid}"]
    row = pd.DataFrame([st.session_state["g_row"]])
    ai = predict_proba_df(row).iloc[0].idxmax()
    ok = guess == ai
    st.session_state["g_last"] = (guess, ai, ok)
    st.session_state["g_rounds"] += 1
    st.session_state["g_done"] = True
    if ok:
        st.session_state["g_score"] += 1
        st.session_state["g_streak"] += 1
    else:
        st.session_state["g_streak"] = 0


for k in ["g_score", "g_rounds", "g_streak"]:
    if k not in st.session_state:
        st.session_state[k] = 0
if "g_row" not in st.session_state:
    new_round()

# ---------------------------------------------------------------- header
head_l, head_r = st.columns([5, 1])
with head_r:
    lang = st.radio("Language", ["English", "हिन्दी"], horizontal=True, label_visibility="collapsed")
hi = lang == "हिन्दी"
with head_l:
    st.markdown(
        "<div class='hero'><h1>🌫️ AQI Predictor</h1>"
        f"<p>{'हवा की क्वालिटी का अंदाज़ा लगाओ, खेलो और सीखो!' if hi else 'Predict air quality, play the game and learn what is in the air we breathe!'}</p></div>",
        unsafe_allow_html=True,
    )

tab_predict, tab_game, tab_whatif, tab_batch, tab_about = st.tabs(
    ["🎯 Predict", "🎮 Play the game", "🔬 What-if", "📂 Batch CSV", "ℹ️ About"]
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
            lo, hi_ = slider_range(f)
            with grid[i % 2]:
                st.slider(
                    LABELS.get(f, f),
                    min_value=lo,
                    max_value=hi_,
                    step=(hi_ - lo) / 200 if hi_ > lo else 1.0,
                    key=f"in_{f}",
                )

    current = pd.DataFrame([{f: st.session_state[f"in_{f}"] for f in features}])
    probs = predict_proba_df(current).iloc[0]
    status = probs.idxmax()
    emoji, msg_en, msg_hi = MASCOT[status]

    with right:
        st.subheader("Prediction")
        st.markdown(
            f"<div class='banner' style='background:{COLORS[status]}'>"
            f"<div class='big'>{emoji}</div><div class='cat'>{status}</div>"
            f"<div class='msg'>{msg_hi if hi else msg_en}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(gauge_svg(probs), unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Confidence" if not hi else "भरोसा", f"{probs.max() * 100:.1f}%")
        if "pm25" in features:
            cigs = st.session_state["in_pm25"] / 22
            m2.metric(
                "🚬 Smoke equivalent" if not hi else "🚬 सिगरेट के बराबर",
                f"≈ {cigs:.1f} /day",
                help="Rough rule of thumb: about 22 µg/m³ of PM2.5 over a day is like smoking 1 cigarette. "
                "It only looks at PM2.5 and is just an estimate.",
            )
        st.info(ADVICE[status][1 if hi else 0])
        st.write("**Probability of each category**")
        st.bar_chart(probs)

# ================= TAB 2: GAME =================
with tab_game:
    st.subheader("🎮 Can you think like the AI?")
    st.write("You get a random day of readings. Guess the AQI category, then see if the AI agrees.")
    s1, s2, s3 = st.columns(3)
    s1.metric("Score", f"{st.session_state['g_score']} / {st.session_state['g_rounds']}")
    s2.metric("Streak 🔥", st.session_state["g_streak"])
    acc = (
        100 * st.session_state["g_score"] / st.session_state["g_rounds"]
        if st.session_state["g_rounds"]
        else 0
    )
    s3.metric("Match rate", f"{acc:.0f}%")

    gid = st.session_state["g_id"]
    row = st.session_state["g_row"]
    st.write("**Today's readings**")
    gcols = st.columns(len(pollutants))
    for col, f in zip(gcols, pollutants):
        col.metric(LABELS.get(f, f), f"{row[f]:.1f}")
    st.caption(
        "  •  ".join(f"{LABELS.get(f, f)}: {row[f]:.1f}" for f in WEATHER if f in row)
    )

    st.radio("Your guess", ORDER, horizontal=True, key=f"guess_{gid}", disabled=st.session_state["g_done"])
    if not st.session_state["g_done"]:
        st.button("🔒 Lock my answer", on_click=lock_answer, type="primary")
    else:
        guess, ai, ok = st.session_state["g_last"]
        if ok:
            st.success(f"✅ Correct! You and the AI both said **{ai}**.")
            if not st.session_state["g_celebrated"]:
                st.balloons()
                st.session_state["g_celebrated"] = True
        else:
            st.error(f"❌ You said **{guess}**, the AI said **{ai}**.")
        st.button("➡️ Next round", on_click=new_round, type="primary")

# ================= TAB 3: WHAT-IF =================
with tab_whatif:
    st.subheader("What happens if one value changes?")
    st.write("Pick a reading. Everything else stays as set in the Predict tab.")
    chosen = st.selectbox("Reading to vary", features, format_func=lambda f: LABELS.get(f, f))
    lo, hi_ = slider_range(chosen)
    sweep_vals = [lo + (hi_ - lo) * i / 49 for i in range(50)]
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

# ================= TAB 4: BATCH =================
with tab_batch:
    st.subheader("Predict many rows at once")
    st.write("Upload a CSV with these columns:")
    st.code(", ".join(features))
    template = pd.DataFrame([PRESETS["😐 Typical day"]] * 3)
    st.download_button("Download sample CSV", template.to_csv(index=False), "sample_input.csv", "text/csv")
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
            st.download_button("Download results", data.to_csv(index=False), "aqi_predictions.csv", "text/csv")

# ================= TAB 5: ABOUT =================
with tab_about:
    st.subheader("About the model")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{meta['accuracy']:.3f}")
    c2.metric("Macro F1", f"{meta['macro_f1']:.3f}")
    st.write("**Feature importance**")
    st.bar_chart(pd.Series(meta["importance"]).sort_values())
    st.write("**Per-class report**")
    st.dataframe(pd.DataFrame(meta["report"]).T.round(3), use_container_width=True)
    st.caption("Built with Python, scikit-learn and Streamlit.")
