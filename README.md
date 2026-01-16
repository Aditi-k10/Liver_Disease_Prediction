# Liver Disease Classification Using Machine Learning

## Project Overview
This project focuses on predicting liver disease conditions using machine learning classification techniques.  
The objective is to classify patients into different liver disease categories based on clinical and biochemical parameters obtained from blood and urine analysis.

Multiple machine learning models were developed and evaluated, and the **Gradient Boosting Classifier** was selected as the final model for deployment due to its better performance.

The final model is deployed as an interactive web application using **Streamlit**, allowing users to enter patient details and obtain real-time predictions.



## Business Objective
The target variable to be predicted is **categorical**, making this a **classification problem**.

The goal of this project is to accurately classify patients into one of the following liver disease categories:
- No Disease
- Suspect Disease
- Hepatitis C
- Fibrosis
- Cirrhosis



## Dataset Description
- **Number of Instances (Rows):** 615  
- **Number of Variables (Columns):** 13  
- **Domain:** Healthcare / Medical Data  

Most input features are numerical. Only one feature (`Sex`) is binary.  
The dataset consists of laboratory measurements related to liver and kidney functions.



## Target Variable
**Category (Diagnosis):**
- `no_disease`
- `suspect_disease`
- `hepatitis_c`
- `fibrosis`
- `cirrhosis`



## Input Features Description

- **Age:** Range 0–100 (normal values vary with age)
- **Sex:** Male or Female
- **Albumin:** Normal range 34–54 g/L  
  Low levels may indicate liver disease such as hepatitis or cirrhosis
- **Alkaline Phosphatase:** Normal range 40–129 U/L  
  Elevated levels indicate liver damage
- **Alanine Aminotransferase (ALT):** Normal range 7–55 U/L  
  High levels indicate liver cell damage
- **Aspartate Aminotransferase (AST):** Normal range 8–48 U/L  
  Elevated values suggest liver disease
- **Bilirubin:** Normal range 1–12 mg/L  
  High levels indicate liver dysfunction
- **Cholinesterase:** Normal range 8–18 U/L  
  Low levels are associated with liver and renal disease
- **Cholesterol:** Less than 5.2 mmol/L  
  High levels are linked with cardiovascular and metabolic disorders
- **Creatinina:**  
  - Male: 61.9–114.9 µmol/L  
  - Female: 53–97.2 µmol/L  
  Abnormal levels indicate kidney dysfunction
- **Gamma Glutamyl Transferase (GGT):** 0–30/50 IU/L  
  Elevated levels indicate liver damage
- **Protein:** Less than 80 mg  
  High levels in urine may indicate kidney or liver problems


## Machine Learning Approach

- This project is formulated as a multi-class classification problem.
- Multiple machine learning classification models were trained and evaluated to determine the best performing algorithm.
- The dataset used in this project is imbalanced, with certain liver disease categories having fewer samples than others.
- Gradient Boosting Classifier was selected as the final model because it handles imbalanced data effectively by focusing more on minority and hard-to-classify cases.
- Gradient Boosting improves model performance by iteratively learning from previous errors and reducing bias toward majority classes.


## Streamlit Web Application
The deployed Streamlit application provides:
- User-friendly data input interface
- Real-time liver disease prediction
- Probability scores for each disease category
- Clear visual feedback for prediction results

## Deployment
The application is deployed using Streamlit Cloud.
Live Application:
https://liverdiseaseprediction-npkixarr2hxnu6earjedhj.streamlit.app/
