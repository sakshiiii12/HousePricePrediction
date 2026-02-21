# Import Library
import streamlit as st
import pandas as pd
import joblib

# Page Confu=iguration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
)

# Load Model
model = joblib.load('model/house_price_model.pkl')

st.title("🏠 House Price Prediction")

st.markdown("### Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=0)
    bedrooms = st.number_input("Bedrooms", min_value=0)
    bathrooms = st.number_input("Bathrooms", min_value=0)
    stories = st.number_input("Stories", min_value=0)
    parking = st.number_input("Parking Spaces", min_value=0)

with col2:
    mainroad = st.selectbox("Main Road Access", ["yes", "no"])
    guestroom = st.selectbox("Guest Room", ["yes", "no"])
    basement = st.selectbox("Basement", ["yes", "no"])
    hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
    airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
    prefarea = st.selectbox("Preferred Area", ["yes", "no"])
    furnishingstatus = st.selectbox(
        "Furnishing Status",
        ["furnished", "semi-furnished", "unfurnished"]
    )

input_df = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "mainroad": [mainroad],
    "guestroom": [guestroom],
    "basement": [basement],
    "hotwaterheating": [hotwaterheating],
    "airconditioning": [airconditioning],
    "parking": [parking],
    "prefarea": [prefarea],
    "furnishingstatus": [furnishingstatus]
})

if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated House Price: ₹ {prediction:,.2f}")