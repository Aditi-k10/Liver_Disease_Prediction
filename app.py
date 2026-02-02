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
# Custom CSS
# ------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #C1F0F6, #FDE2F3);
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }

    .prediction-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 3px 3px 20px rgba(0,0,0,0.15);
        text-align: center;
        margin-top: 20px;
    }

    h1 { color: #FF4B4B; }
    h2 { color: #FF6F61; }

    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        background-color: #FF4B4B;
        color: white;
        font-size: 18px;
        padding: 10px 30px;
        border-radius: 10px;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Load Model
# ------------------------------
with open("gradient_boosting_model.pkl", "rb") as file:
    model = pickle.load(file)

# ------------------------------
# Page Header (NO WHITE BAR)
# ------------------------------
st.markdown("<h1 style='text-align:center;'>🩺 Liver Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#5A5A5A; font-size:18px;'>Enter patient clinical details below</p>",
    unsafe_allow_html=True
)

# ------------------------------
# Input Fields (2 Columns)
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=0)
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

with col2:
    bilirubin = st.number_input("Bilirubin", min_value=0.0, value=0.0)
    cholinesterase = st.number_input("Cholinesterase", min_value=0.0, value=0.0)
    cholesterol = st.number_input("Cholesterol", min_value=0.0, value=0.0)
    creatinina = st.number_input("Creatinina", min_value=0.0, value=0.0)
    gamma_glutamyl_transferase = st.number_input("Gamma Glutamyl Transferase", min_value=0.0, value=0.0)
    protein = st.number_input("Protein", min_value=0.0, value=0.0)

if st.button("Predict"):
    if sex_encoded is None:
        st.warning("⚠️ Please select Sex")
    else:
        input_data = np.array([[
            age, sex_encoded, albumin, alkaline_phosphatase,
            alanine_aminotransferase, aspartate_aminotransferase,
            bilirubin, cholinesterase, cholesterol, creatinina,
            gamma_glutamyl_transferase, protein
        ]])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)

        # Prediction text
        st.markdown("<h2>Disease Predicted</h2>", unsafe_allow_html=True)

        if prediction == "no_disease":
            st.success("✅ No Liver Disease")
        elif prediction == "suspect_disease":
            st.warning("⚠️ Suspected Liver Disease")
        else:
            st.error("❌ Liver Disease Detected")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Probability of Each Condition")

        # Compact probability bars
        for cls, prob in zip(model.classes_, probabilities):
            if cls == "no_disease":
                color = "#2ECC71"
            elif cls == "suspect_disease":
                color = "#F1C40F"
            else:
                color = "#E74C3C"

            st.markdown(
                f"""
                <div class="prob-bar-bg">
                    <div class="prob-bar" style="width:{prob*100}%; background-color:{color};">
                        {cls.replace('_',' ').title()} : {prob*100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Footer Space
# ------------------------------
st.markdown("<br>", unsafe_allow_html=True)

