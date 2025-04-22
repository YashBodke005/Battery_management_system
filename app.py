import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd

# ======================= #
#    Page Configuration   #
# ======================= #
st.set_page_config(
    page_title="🔋 EV Battery RUL Predictor",
    page_icon="⚡",
    layout="centered"
)

st.title("🔋 Battery Health Monitoring App")
st.markdown("""
Welcome to the **EV Battery Remaining Useful Life (RUL)** Predictor!  
Input your battery parameters below and let the model estimate how much life your battery has left.  
""")

st.markdown("---")

# ======================= #
#     Load Model/Scaler   #
# ======================= #
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("models/random_forest.joblib")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# ======================= #
#    Define Input Fields  #
# ======================= #
feature_names = [
    
    "Discharge Time (s)",
    "Decrement 3.6-3.4V (s)",
    "Max. Voltage Dischar. (V)",
    "Min. Voltage Charg. (V)",
    "Time at 4.15V (s)",
    "Time constant current (s)",
    "Charging time (s)"
]

default_vals = [500, 1500, 450, 3.9, 3.6, 3000, 1200, 1800]

user_input = []
with st.form("rul_form"):
    for name, default in zip(feature_names, default_vals):
        val = st.number_input(f"{name}:", value=float(default), format="%.2f")
        user_input.append(val)

    submitted = st.form_submit_button("🔮 Predict RUL")

# ======================= #
#     Predict & Display   #
# ======================= #
if submitted:
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    predicted_rul = model.predict(input_scaled)[0]

    st.success(f"✅ Estimated Remaining Useful Life: **{predicted_rul:.2f} cycles**")
    st.balloons()

    st.markdown("### 📋 Input Summary")
    st.dataframe(pd.DataFrame([user_input], columns=feature_names))

# ======================= #
#       Footer            #
# ======================= #
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center;'>
        <sub>Created with ❤️ for E-Vehicle Battery Monitoring</sub><br>
        
    </div>
    """, unsafe_allow_html=True
)
