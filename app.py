import streamlit as st
import pickle
import numpy as np

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🩺",
    layout="centered"
)

# ------------------------------
# Custom CSS
# ------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #E3FDFD, #FFE6FA);
    font-family: 'Segoe UI', sans-serif;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.1);
    margin-bottom: 25px;
}

h1 {
    text-align: center;
    color: #E74C3C;
}

h3 {
    text-align: center;
}

.stButton>button {
    background-color: #E74C3C;
    color: white;
    font-size: 16px;
    padding: 10px 28px;
    border-radius: 10px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load model
# ------------------------------
with open("gradient_boosting_model.pkl", "rb") as file:
    model = pickle.load(file)

# ------------------------------
# Header
# ------------------------------
st.markdown("<h1>🩺 Liver Disease Prediction</h1>", unsafe_allow_html=True)

# ------------------------------
# Input Section
# ------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120)
    sex = st.selectbox("Sex", ["Female", "Male"])
    sex_encoded = 0 if sex == "Female" else 1

    albumin = st.number_input("Albumin")
    alkaline_phosphatase = st.number_input("Alkaline Phosphatase")
    alt = st.number_input("Alanine Aminotransferase (SGPT)")
    ast = st.number_input("Aspartate Aminotransferase (SGOT)")

with col2:
    bilirubin = st.number_input("Bilirubin")
    cholinesterase = st.number_input("Cholinesterase")
    cholesterol = st.number_input("Cholesterol")
    creatinina = st.number_input("Creatinina")
    ggt = st.number_input("Gamma Glutamyl Transferase")
    protein = st.number_input("Protein")

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("Predict")
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Prediction
# ------------------------------
if predict_btn:
    input_data = np.array([[age, sex_encoded, albumin, alkaline_phosphatase,
                             alt, ast, bilirubin, cholinesterase,
                             cholesterol, creatinina, ggt, protein]])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prediction Result")

    if prediction == "no_disease":
        st.success("✅ No Liver Disease Detected")
    elif prediction == "suspect_disease":
        st.warning("⚠️ Suspected Liver Disease")
    else:
        st.error("❌ Liver Disease Detected")

    st.markdown("### Confidence Score")

    for cls, prob in zip(model.classes_, probabilities):
        st.progress(prob)
        st.write(f"**{cls.replace('_',' ').title()}** : {prob*100:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)
