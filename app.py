"""Interactive demo:  crop + location + season  ->  likelihood of a good yield.

Run:  streamlit run app.py
"""
import sys, json, pathlib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import outlook

st.set_page_config(page_title="Crop Yield Outlook", page_icon="🌾", layout="centered")
st.title("🌾 Crop Yield Outlook")
st.caption("Pick a crop, a place and a season. The model estimates how likely a Good, Average or Poor yield is.")


@st.cache_resource
def load():
    return outlook.YieldOutlook()


yo = load()
opt = yo.options()

c1, c2 = st.columns(2)
crop = c1.selectbox("1 · Crop", opt["crops"], index=opt["crops"].index("Maize"))
season_of_crop = opt["crop_season"][crop]
season = c2.selectbox("3 · Season", opt["seasons"], index=opt["seasons"].index(season_of_crop))
c3, c4 = st.columns(2)
state = c3.selectbox("2 · State", sorted(opt["locations"]))
district = c4.selectbox("District", opt["locations"][state])

with st.expander("Season conditions (optional)"):
    mode = st.radio("What should the model assume about this season?",
                    ["Typical season", "Drier than usual", "Wetter than usual", "Replay a past year"], horizontal=True)
    year = st.slider("Past year", 2001, 2017, 2015) if mode == "Replay a past year" else None
scenario = {"Typical season": "normal", "Drier than usual": "dry", "Wetter than usual": "wet"}.get(mode, "normal")

loss, detected = 0.0, None
if crop == "Maize":
    with st.expander("Leaf photo (optional, maize only)"):
        up = st.file_uploader("Upload a maize leaf photo to check for disease", type=["jpg", "jpeg", "png"])
        share = st.slider("Share of scouted leaves that look like this", 0, 100, 40) / 100
        if up is not None:
            import tensorflow as tf
            from tensorflow import keras
            from PIL import Image
            m = keras.models.load_model(ROOT / "models" / "leaf_disease_mobilenetv2.keras")
            names = json.load(open(ROOT / "models" / "leaf_disease_classes.json"))
            img = Image.open(up).convert("RGB").resize((224, 224))
            p = m.predict(np.expand_dims(np.array(img, dtype="float32"), 0), verbose=0)[0]
            detected = names[int(p.argmax())]
            st.image(img, width=220, caption=f"{detected} ({p.max():.0%} confident)")
            sev = {"Gray leaf spot": .25, "Northern leaf blight": .20, "Common rust": .08, "Healthy": 0.0}   # assumptions
            loss = sev.get(detected, 0) * share

try:
    r = yo.predict(crop=crop, state=state, district=district, season=season, year=year, scenario=scenario, yield_loss=loss)
except ValueError as e:
    st.error(str(e)); st.stop()

st.subheader(f"Most likely: {r['verdict']} yield")
a, b, c = st.columns(3)
a.metric("Good", f"{r['p_good']:.0%}"); b.metric("Average", f"{r['p_average']:.0%}"); c.metric("Poor", f"{r['p_poor']:.0%}")
st.bar_chart(pd.DataFrame({"probability": [r["p_poor"], r["p_average"], r["p_good"]]}, index=["Poor", "Average", "Good"]))
st.write(f"**Expected yield:** about {r['expected_yield']:,.0f} kg/ha (80 % range {r['yield_low']:,.0f}–{r['yield_high']:,.0f}). "
         f"“Good” means above roughly {r['good_threshold']:,.0f} kg/ha for {crop.lower()}.")
st.write(f"**Based on:** {r['conditions']}.")
if r["drivers"]:
    st.write("**Why:** " + "; ".join(r["drivers"]) + ".")
if detected:
    st.info(f"Leaf evidence: {detected} on {share:.0%} of leaves → assumed yield loss {loss:.0%}, already included above.")
st.caption("Research prototype built on 2001-2017 district-level data; not agronomic advice.")
