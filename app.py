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
st.markdown("<h1 style='text-align:center;'>🩺 Liver Disease Prediction</h1>", unsafe_allow_html=True)
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
        # Disease detected (name in front)
        # ------------------------------
        st.markdown("## Disease Detected")

        disease_map = {
            "no_disease": "No Liver Disease",
            "suspect_disease": "Suspected Liver Disease",
            "liver_disease": "Liver Disease Detected"
        }

        detected_disease = disease_map.get(pred, pred.replace("_", " ").title())

        if pred == "no_disease":
            st.success(f"✅ {detected_disease}")
        elif pred == "suspect_disease":
            st.warning(f"⚠️ {detected_disease}")
        else:
            st.error(f"❌ {detected_disease}")

        # ------------------------------
        # Probability of each disease
        # ------------------------------
        bars_html = """
        <div style="background:white;padding:20px;border-radius:12px;
                    box-shadow:0 4px 12px rgba(0,0,0,0.15);margin-top:15px;">
        <h3>Probability of Each Disease</h3>
        """

        for cls, p in zip(model.classes_, probs):
            if cls == "no_disease":
                color = "#2ECC71"
            elif cls == "suspect_disease":
                color = "#F1C40F"
            else:
                color = "#E74C3C"

            bars_html += f"""
            <div style="margin-bottom:12px;">
                <strong>{cls.replace('_',' ').title()}</strong>
                <div style="background:#E0E0E0;border-radius:8px;height:28px;">
                    <div style="width:{p*100:.1f}%;
                                background:{color};
                                height:28px;
                                border-radius:8px;
                                text-align:center;
                                color:white;
                                font-weight:600;
                                line-height:28px;">
                        {p*100:.2f}%
                    </div>
                </div>
            </div>
            """

        bars_html += "</div>"

        components.html(bars_html, height=260)


