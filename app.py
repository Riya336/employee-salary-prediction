import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (these must match your training feature columns)
st.sidebar.header("Input Employee Details")
education_encoder = joblib.load("encoders/education_encoder.pkl")  # Load your encoders
occupation_encoder = joblib.load("encoders/occupation_encoder.pkl")
gender_encoder = joblib.load("encoders/gender_encoder.pkl")
workclass_encoder = joblib.load("encoders/workclass_encoder.pkl")
native_country_encoder = joblib.load("encoders/native_country_encoder.pkl")
education_levels = education_encoder.classes_
occupation_titles = occupation_encoder.classes_
gender_titles = gender_encoder.classes_
workclass_titles = workclass_encoder.classes_
native_country_titles = native_country_encoder.classes_

# ✨ Replace these fields with your dataset's actual input columns
age = st.sidebar.slider("Age", 18, 65, 30)
education=st.sidebar.selectbox("Education Level", education_levels)
occupation=st.sidebar.selectbox("Occupation", occupation_titles)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
gender= st.sidebar.selectbox("Gender", gender_titles)
workclass= st.sidebar.selectbox("Work Class", workclass_titles)
native_country= st.sidebar.selectbox("Native Country", native_country_titles)
education = education_encoder.transform([education])[0]
occupation = occupation_encoder.transform([occupation])[0]
gender = gender_encoder.transform([gender])[0]
workclass = workclass_encoder.transform([workclass])[0]
native_country = native_country_encoder.transform([native_country])[0]

# Build input DataFrame (⚠️ must match preprocessing of your training data)
input_df = pd.DataFrame([[age, workclass, education, occupation, gender, hours_per_week, native_country]],
                        columns=['age', 'workclass', 'education', 'occupation', 'gender', 'hours-per-week', 'native-country'])

st.write("### 🔎 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')