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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Load the trained Gradient Boosting model
# ------------------------------
with open("gradient_boosting_model.pkl", "rb") as file:
    model = pickle.load(file)

# ------------------------------
# Page Header
# ------------------------------
st.markdown(
    """
    <div style='text-align: center; background-color:#FFDDC1; padding:15px; border-radius:10px;'>
        <h1 style='color:#FF4B4B;'>🩺 Liver Disease Prediction App</h1>
        <p style='color:#5A5A5A; font-size:18px;'>Enter patient details below to predict liver condition</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------
# Input Form: 2 Columns (6 features each)
# ------------------------------
with st.container():
    col1, col2 = st.columns(2)

    with col1:
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

    with col2:
        bilirubin = st.number_input("Bilirubin", min_value=0.0, value=0.0)
        cholinesterase = st.number_input("Cholinesterase", min_value=0.0, value=0.0)
        cholesterol = st.number_input("Cholesterol", min_value=0.0, value=0.0)
        creatinina = st.number_input("Creatinina", min_value=0.0, value=0.0)
        gamma_glutamyl_transferase = st.number_input("Gamma Glutamyl Transferase", min_value=0.0, value=0.0)
        protein = st.number_input("Protein", min_value=0.0, value=0.0)

# ------------------------------
# Predict button
# ------------------------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Predict", type="primary"):
    if sex_encoded is None:
        st.warning("⚠️ Please select Sex")
    else:
        # Prepare input data
        input_data = np.array([[age, sex_encoded, albumin, alkaline_phosphatase,
                                alanine_aminotransferase, aspartate_aminotransferase,
                                bilirubin, cholinesterase, cholesterol, creatinina,
                                gamma_glutamyl_transferase, protein]])

        # Predict class & probability
        predicted_class = model.predict(input_data)[0]
        proba_all = model.predict_proba(input_data)[0]

        # ------------------------------
        # Display results in a colorful card
        # ------------------------------
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center; color:#FF4B4B;'>Prediction Result</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            if predicted_class == "no_disease":
                st.success(f"✅ Predicted Class: {predicted_class.replace('_', ' ').title()}")
            elif predicted_class == "suspect_disease":
                st.warning(f"⚠️ Predicted Class: {predicted_class.replace('_', ' ').title()}")
            else:
                st.error(f"❌ Predicted Class: {predicted_class.replace('_', ' ').title()}")

        with col2:
            st.subheader("Confidence (%)")
            for cls, prob in zip(model.classes_, proba_all):
                # Colored progress bars
                if cls == "no_disease":
                    color = "#2ECC71"  # Green
                elif cls == "suspect_disease":
                    color = "#F1C40F"  # Yellow
                else:
                    color = "#E74C3C"  # Red

                st.markdown(f"""
                <div style='background-color:#E0E0E0; border-radius:5px; padding:3px; margin-bottom:5px;'>
                    <div style='width:{prob*100}%; background-color:{color}; padding:5px; border-radius:5px; color:white; font-weight:bold; text-align:center;'>
                        {cls.replace('_',' ').title()}: {prob*100:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Designed with ❤️ for liver health monitoring</p>", unsafe_allow_html=True)
