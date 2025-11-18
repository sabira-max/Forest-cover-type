import streamlit as st
import pandas as pd
import pickle
import numpy as np


# ============================
# 🎨 CUSTOM AMAZON-STYLE THEME
# ============================
amazon_css = """
<style>

    /* Main background */
    .main {
        background-color: #f3e5ab !important;   /* Soft Amazon yellow */
    }

    /* Title */
    h1 {
        color: #232f3e !important;               /* Amazon dark blue */
        font-weight: 800 !important;
    }

    /* Section headers */
    h2, h3, h4 {
        color: #131921 !important;               /* Amazon navbar black */
    }

    /* Boxes and form inputs */
    .stSlider, .stNumberInput, .stSelectbox, .stTextInput {
        background-color: #fff4cc !important;
        padding: 12px !important;
        border-radius: 10px !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #ff9900 !important;     /* Amazon orange */
        color: black !important;
        font-size: 16px;
        font-weight: 700;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #ffb84d !important;
        color: black !important;
    }

    /* Result box */
    .st-success {
        background-color: #ffdd99 !important;
        padding: 15px;
        border-radius: 10px;
        color: #232f3e !important;
        font-size: 18px;
        font-weight: 700;
    }

</style>
"""

st.markdown(amazon_css, unsafe_allow_html=True)

# ============================================================
# 1️⃣ Load model & preprocessing objects
# ============================================================
@st.cache_resource
def load_artifacts():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("Processed_Data.pkl", "rb") as f:
        processed = pickle.load(f)

    return model, processed["scaler"], processed["selected_columns"]

model, scaler, selected_features = load_artifacts()

# ===================== COVER TYPE NAMES ======================
cover_type_names = {
    0: "Spruce/Fir",
    1: "Lodgepole Pine",
    2: "Ponderosa Pine",
    3: "Cottonwood/Willow",
    4: "Aspen",
    5: "Douglas-fir",
    6: "Krummholz"
}

# ============================================================
# 2️⃣ Streamlit Setup
# ============================================================
st.set_page_config(page_title="Forest Cover Type Prediction", layout="centered")
st.title("🌲Forest Cover Type Prediction🌲")
st.write("Predict forest cover type based on 15 terrain & soil features.")

st.markdown("---")

# ============================================================
# 3️⃣ INPUTS FOR ALL 15 FEATURES (Updated)
# ============================================================
st.subheader("🌲 Input Features")

# ---- 3 MAIN FEATURES AS SLIDERS ----
Elevation = st.slider("Elevation", 1000, 4000, 2500)
HD_Hydro = st.slider("Horizontal_Distance_To_Hydrology", 0, 6000, 1000)
HD_Roads = st.slider("Horizontal_Distance_To_Roadways", 0, 6000, 1000)

st.write("### 🌲 Additional Features")

# ---- OTHER NUMERIC FEATURES AS NORMAL INPUTS ----
HD_Fire = st.number_input("Horizontal_Distance_To_Fire_Points", 0, 7000, 1000)
VD_Hydro = st.number_input("Vertical_Distance_To_Hydrology", -500, 500, 30)
Aspect = st.number_input("Aspect", 0, 360, 180)
Hillshade_3pm = st.number_input("Hillshade_3pm", 0, 255, 200)
Hillshade_Noon = st.number_input("Hillshade_Noon", 0, 255, 220)
Slope = st.number_input("Slope", 0, 70, 10)

# ---- SOIL TYPES AS SELECTBOX (0/1) ----
st.subheader("🌲 Soil Type (0 = No, 1 = Yes)")

Soil_29 = st.selectbox("Soil_Type_29", [0, 1])
Soil_25 = st.selectbox("Soil_Type_25", [0, 1])
Soil_27 = st.selectbox("Soil_Type_27", [0, 1])
Soil_26 = st.selectbox("Soil_Type_26", [0, 1])
Soil_21 = st.selectbox("Soil_Type_21", [0, 1])
Soil_24 = st.selectbox("Soil_Type_24", [0, 1])

# ============================================================
# 4️⃣ Build Input Row (15 Features)
# ============================================================
input_data = {
    'Elevation': Elevation,
    'Horizontal_Distance_To_Roadways': HD_Roads,
    'Horizontal_Distance_To_Fire_Points': HD_Fire,
    'Horizontal_Distance_To_Hydrology': HD_Hydro,
    'Vertical_Distance_To_Hydrology': VD_Hydro,
    'Aspect': Aspect,
    'Hillshade_3pm': Hillshade_3pm,
    'Hillshade_Noon': Hillshade_Noon,
    'Slope': Slope,
    'Soil_Type_29': Soil_29,
    'Soil_Type_25': Soil_25,
    'Soil_Type_27': Soil_27,
    'Soil_Type_26': Soil_26,
    'Soil_Type_21': Soil_21,
    'Soil_Type_24': Soil_24
}

input_df = pd.DataFrame([input_data])

# ============================================================
# 5️⃣ Scale Numeric Features Only
# ============================================================
numeric_features = scaler.feature_names_in_
categorical_features = [col for col in selected_features if col not in numeric_features]

numeric_scaled = scaler.transform(input_df[numeric_features])
numeric_scaled_df = pd.DataFrame(numeric_scaled, columns=numeric_features)

categorical_df = input_df[categorical_features].reset_index(drop=True)

final_input = pd.concat([numeric_scaled_df, categorical_df], axis=1)
final_input = final_input[selected_features]

st.caption("✅ All 15 features processed & scaled correctly.")

# ============================================================
# 6️⃣ Predict Cover Type
# ============================================================
if st.button("Predict Cover Type"):
    pred = model.predict(final_input)[0]
    cover_name = cover_type_names.get(pred, "Unknown")

    st.success(f"🌳 Predicted Forest Cover Type 🌳 : **{cover_name}** (Class {pred})")

    # Optional probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(final_input)[0]
        st.write("### 🔎 Class Probabilities 🌲")
        st.write(pd.DataFrame({
            "Class": list(cover_type_names.values()),
            "Probability": probs
        }))

# ======================================================
# 8️⃣ SAVE FINAL MODEL + SCALER + FEATURES FOR STREAMLIT
# ======================================================

processed = {
    "scaler": scaler,
    "selected_columns": selected_features
}

# Save preprocessing
with open("Processed_Data.pkl", "wb") as f:
    pickle.dump(processed, f)

# Save trained model
with open("best_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Successfully saved: Processed_Data.pkl + best_model.pkl")

