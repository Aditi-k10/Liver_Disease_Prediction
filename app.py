# app.py
import streamlit as st
import pickle
import numpy as np

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# ------------------------------
# Custom CSS for background & cards
# ------------------------------
st.markdown(
    """
    <style>
    /* Gradient background */
    .stApp {
        background: linear-gradient(to right, #FFDDE1, #FFDEE9);
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Input cards */
    .input-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    /* Prediction card */
    .prediction-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 3px 3px 20px rgba(0,0,0,0.15);
        text-align: center;
        margin-top: 20px;
    }

    h1 {
        color: #FF4B4B;
    }

    h2 {
        color: #FF6F61;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Load the trained model
# ------------------------------
with open("gradient_boosting_model.pkl", "rb") as file:
    model = pickle.load(file)

# ------------------------------
# Page Header
# ------------------------------
st.markdown(
    """
    <div style='text-align:center; margin-bottom:30px;'>
        <h1>🩺 Liver Disease Prediction App</h1>
        <p style='color:#5A5A5A; font-size:18px;'>Enter patient details below to get prediction</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Input Form inside a card
# ------------------------------
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='input-card'>", unsafe_allow_html=True)
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        sex = st.selectbox("Sex", ("Select", "Female", "Male"))
        if sex == "Female":
            sex_encoded = 0
        elif sex == "Male":
            sex_encoded = 1
        else:
            sex_encoded = None

        albumin = st.number_input("Albumin", min_value=0.0, value=0.0)
        alkaline_phosphatase = st.number_input("Alkaline Phosphatase", min_value=0.0, value=0.0)
        alanine_aminotransferase = st.number_input("Alanine Aminotransferase (SGPT)", min_value=0.0, value=0.0)
        aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (SGOT)", min_value=0.0, value=0.0)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='input-card'>", unsafe_allow_html=True)
        bilirubin = st.number_input("Bilirubin", min_value=0.0, value=0.0)
        cholinesterase = st.number_input("Cholinesterase", min_value=0.0, value=0.0)
        cholesterol = st.number_input("Cholesterol", min_value=0.0, value=0.0)
        creatinina = st.number_input("Creatinina", min_value=0.0, value=0.0)
        gamma_glutamyl_transferase = st.number_input("Gamma Glutamyl Transferase", min_value=0.0, value=0.0)
        protein = st.number_input("Protein", min_value=0.0, value=0.0)
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Predict Button
# ------------------------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Predict", type="primary"):
    if sex_encoded is None:
        st.warning("⚠️ Please select Sex")
    else:
        input_data = np.array([[age, sex_encoded, albumin, alkaline_phosphatase,
                                alanine_aminotransferase, aspartate_aminotransferase,
                                bilirubin, cholinesterase, cholesterol, creatinina,
                                gamma_glutamyl_transferase, protein]])
        predicted_class = model.predict(input_data)[0]
        proba_all = model.predict_proba(input_data)[0]

        # ------------------------------
        # Prediction Card
        # ------------------------------
        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)

        st.markdown(f"<h2>Prediction Result</h2>", unsafe_allow_html=True)
        if predicted_class == "no_disease":
            st.markdown("<h3 style='color:#2ECC71;'>✅ No Disease</h3>", unsafe_allow_html=True)
        elif predicted_class == "suspect_disease":
            st.markdown("<h3 style='color:#F1C40F;'>⚠️ Suspect Disease</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:#E74C3C;'>❌ Liver Disease Detected</h3>", unsafe_allow_html=True)

        st.markdown("<br><h4>Confidence of Each Class</h4>", unsafe_allow_html=True)
        for cls, prob in zip(model.classes_, proba_all):
            if cls == "no_disease":
                color = "#2ECC71"
            elif cls == "suspect_disease":
                color = "#F1C40F"
            else:
                color = "#E74C3C"

            st.markdown(f"""
            <div style='background-color:#E0E0E0; border-radius:10px; margin-bottom:5px;'>
                <div style='width:{prob*100}%; background-color:{color}; padding:8px; border-radius:10px; color:white; font-weight:bold; text-align:center;'>
                    {cls.replace('_',' ').title()}: {prob*100:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#5A5A5A;'>Designed with ❤️ for liver health monitoring</p>", unsafe_allow_html=True)
