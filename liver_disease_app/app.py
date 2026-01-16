# app.py
import streamlit as st
import pickle
import numpy as np 


# Load the trained Gradient Boosting model

with open("gradient_boosting_model.pkl", "rb") as file:
    model = pickle.load(file)

# App title

st.title("Liver Disease Prediction")
st.write("Fill the patient details below to predict liver condition")


# User Inputs

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
bilirubin = st.number_input("Bilirubin", min_value=0.0, value=0.0)
cholinesterase = st.number_input("Cholinesterase", min_value=0.0, value=0.0)
cholesterol = st.number_input("Cholesterol", min_value=0.0, value=0.0)
creatinina = st.number_input("Creatinina", min_value=0.0, value=0.0)
gamma_glutamyl_transferase = st.number_input("Gamma Glutamyl Transferase", min_value=0.0, value=0.0)
protein = st.number_input("Protein", min_value=0.0, value=0.0)


# Prediction

if st.button("Predict"):

    if sex_encoded is None:
        st.warning("Please select Sex")
    else:
        # Prepare input data in correct order
        input_data = np.array([[
            age,
            sex_encoded,
            albumin,
            alkaline_phosphatase,
            alanine_aminotransferase,
            aspartate_aminotransferase,
            bilirubin, 
            cholinesterase,
            cholesterol,
            creatinina,
            gamma_glutamyl_transferase,
            protein
        ]])

        # Predict the class
        predicted_class = model.predict(input_data)[0]

        # Predict probabilities for all classes
        proba_all = model.predict_proba(input_data)[0]

        # Display predicted class 
        st.subheader("Predicted Class")
        if predicted_class == "no_disease":
            st.success(f"Predicted Class: {predicted_class.replace('_', ' ').title()}")
        elif predicted_class == "suspect_disease":
            st.warning(f"Predicted Class: {predicted_class.replace('_', ' ').title()}")
        else:
            st.error(f"Predicted Class: {predicted_class.replace('_', ' ').title()}")

        # Display probability for all classes
        st.subheader("Probability of Each Class")
        for cls, prob in zip(model.classes_, proba_all):
            st.write(f"{cls.replace('_', ' ').title()}: {prob*100:.2f}%")
