# app.py
import streamlit as st
import pickle
import numpy as np
import streamlit.components.v1 as components

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# ------------------------------
# Global CSS (no white bar)
# ------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #C1F0F6, #FDE2F3);
    font-family: 'Segoe UI', sans-serif;
}
.block-container { padding-top: 1rem; }

h1 { color: #FF4B4B; }
h2 { color: #FF6F61; }

.stButton > button {
    display: block;
    margin: auto;
    background-color: #FF4B4B;
    color: white;
    font-size: 18px;
    padding: 10px 30px;
    border-radius: 10px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load model
# ------------------------------
with open("gradient_boosting_model.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------------------
# Header
# ------------------------------
st.markdown(
    """
    <div style="margin-top:40px; text-align:center;">
        <h1 style="font-size:48px; margin-bottom:10px;">
            🩺 Liver Disease Prediction App
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<p style='text-align:center;'>Enter patient clinical details</p>", unsafe_allow_html=True)

# ------------------------------
# Inputs
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 0, 120, 0)
    sex = st.selectbox("Sex", ("Select", "Female", "Male"))
    sex_encoded = 0 if sex == "Female" else 1 if sex == "Male" else None
    albumin = st.number_input("Albumin", 0.0)
    alkaline_phosphatase = st.number_input("Alkaline Phosphatase", 0.0)
    alt = st.number_input("Alanine Aminotransferase (SGPT)", 0.0)
    ast = st.number_input("Aspartate Aminotransferase (SGOT)", 0.0)

with col2:
    bilirubin = st.number_input("Bilirubin", 0.0)
    cholinesterase = st.number_input("Cholinesterase", 0.0)
    cholesterol = st.number_input("Cholesterol", 0.0)
    creatinina = st.number_input("Creatinina", 0.0)
    ggt = st.number_input("Gamma Glutamyl Transferase", 0.0)
    protein = st.number_input("Protein", 0.0)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict"):
    if sex_encoded is None:
        st.warning("⚠️ Please select Sex")
    else:
        X = np.array([[age, sex_encoded, albumin, alkaline_phosphatase,
                       alt, ast, bilirubin, cholinesterase,
                       cholesterol, creatinina, ggt, protein]])

        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]

        # ------------------------------
        # Disease Detected (same line, red)
        # ------------------------------
        st.markdown(
            f"""
            <h2>
                Disease Detected :
                <span style="color:#E74C3C; font-weight:700;">
                    {pred.replace('_', ' ').title()}
                </span>
            </h2>
            """,
            unsafe_allow_html=True
        )

        # ------------------------------
        # Probability of each disease (TEXT ONLY)
        # ------------------------------
        st.markdown("### Probability of Each Disease")

        for cls, p in zip(model.classes_, probs):
            st.markdown(
                f"""
                <p style="font-size:18px; margin:6px 0;">
                    <strong>{cls.replace('_',' ').title()}</strong> :
                    {p*100:.2f}%
                </p>
                """,
                unsafe_allow_html=True
            )







