import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =================================================
# Load trained artifacts
# =================================================
model = joblib.load("used_car_price_xgboost.pkl")
columns = joblib.load("column.pkl")
brand_mapping = joblib.load("brand_mapping.pkl")

GLOBAL_BRAND_MEAN = brand_mapping.mean()

# =================================================
# Manual encoders (MUST match training)
# =================================================
fuel_type_map = {
    "Petrol": 0,
    "Diesel": 1,
    "Electric": 2,
    "Hybrid": 3,
    "CNG": 4
}

transmission_map = {
    "Manual": 0,
    "Automatic": 1
}

binary_map = {
    "No": 0,
    "Yes": 1
}

accident_map = {
    "No Accident": 0,
    "Minor": 1,
    "Major": 2
}

color_map = {
    "Black": 0,
    "White": 1,
    "Silver": 2,
    "Gray": 3,
    "Red": 4,
    "Blue": 5,
    "Other": 6
}

# =================================================
# Streamlit page config
# =================================================
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Used Car Price Prediction System")
st.write("Predict the **resale price** of a used car using a trained XGBoost model.")
st.divider()

# =================================================
# User Inputs
# =================================================
st.subheader("Enter Car Details")

brand = st.selectbox(
    "Brand",
    sorted(brand_mapping.index.tolist())
)

fuel_type = st.selectbox(
    "Fuel Type",
    list(fuel_type_map.keys())
)

transmission = st.selectbox(
    "Transmission",
    list(transmission_map.keys())
)

ext_col = st.selectbox(
    "Exterior Color",
    list(color_map.keys())
)

int_col = st.selectbox(
    "Interior Color",
    list(color_map.keys())
)

clean_title = st.selectbox(
    "Clean Title",
    ["Yes", "No"]
)

hp = st.number_input(
    "Horsepower (HP)",
    min_value=50, max_value=1500, value=150, step=10
)

engine_disp = st.number_input(
    "Engine Displacement (Liters)",
    min_value=0.5, max_value=10.0, value=2.0, step=0.1
)

is_v_engine = st.selectbox(
    "V Engine?",
    ["Yes", "No"]
)

accident = st.selectbox(
    "Accident History",
    list(accident_map.keys())
)

vehicle_age = st.number_input(
    "Vehicle Age (Years)",
    min_value=0, max_value=50, value=5, step=1
)

mileage_per_year = st.number_input(
    "Mileage Per Year (km)",
    min_value=0, max_value=60000, value=12000, step=1000
)

# =================================================
# Feature engineering (IDENTICAL to training)
# =================================================
brand_encoded = brand_mapping.get(brand, GLOBAL_BRAND_MEAN)

age_mid = 1 if 5 <= vehicle_age < 10 else 0
age_old = 1 if 10 <= vehicle_age < 15 else 0
age_very_old = 1 if vehicle_age >= 15 else 0

mil_med = 1 if 10000 <= mileage_per_year < 15000 else 0
mil_high = 1 if 15000 <= mileage_per_year < 20000 else 0
mil_vhigh = 1 if mileage_per_year >= 20000 else 0

# =================================================
# Prediction
# =================================================
if st.button("🔮 Predict Price"):

    input_data = {
        "fuel_type": fuel_type_map[fuel_type],
        "transmission": transmission_map[transmission],
        "ext_col": color_map[ext_col],
        "int_col": color_map[int_col],
        "clean_title": binary_map[clean_title],
        "hp": hp,
        "engine displacement": engine_disp,
        "is_v_engine": 1 if is_v_engine == "Yes" else 0,
        "Accident_Impact": accident_map[accident],
        "Vehicle_Age": vehicle_age,
        "Mileage_per_Year": mileage_per_year,
        "Age_Mid": age_mid,
        "Age_Old": age_old,
        "Age_Very Old": age_very_old,
        "Milage_Medium": mil_med,
        "Milage_High": mil_high,
        "Milage_Very High": mil_vhigh,
        "brand_encoded": brand_encoded
    }

    input_df = pd.DataFrame([input_data])

    # Enforce exact column order
    input_df = input_df.reindex(columns=columns)

    # Force numeric dtype (XGBoost requirement)
    input_df = input_df.astype(float)

    # Predict log price
    log_price = model.predict(input_df)

    # Convert back to real price
    predicted_price = np.expm1(log_price)[0]

    st.success(f"💰 Estimated Car Price: ₹ {(predicted_price * 90):,.2f}")

    st.caption("Prediction generated using trained XGBoost regression model.")

# =================================================
# Footer
# =================================================
st.divider()
st.caption("XGBoost • Streamlit • Joblib • Production-ready ML App")
