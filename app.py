import streamlit as st
import numpy as np
import joblib

# Load model yang telah dilatih
model = joblib.load('./model/model_rf.joblib')

# Dictionary untuk mapping hasil prediksi ke label
prediction_labels = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}

# Judul prediksi
st.subheader('Jaya-Jaya Institut - Student Dropout Prediction')

# Input data dari pengguna
col1, col2 = st.columns(2)

with col1 :
    curricular_units_1st_sem_approved = st.number_input('Curricular Units 1st Sem Approved', min_value=0, max_value=30, value=8)
    curricular_units_2nd_sem_approved = st.number_input('Curricular Units 2nd Sem Approved', min_value=0, max_value=30, value=12)

with col2 :
    curricular_units_1st_sem_grade = st.number_input('Curricular Units 1st Sem Grade', min_value=0, max_value=20, value=8)
    curricular_units_2nd_sem_grade = st.number_input('Curricular Units 2nd Sem Grade', min_value=0, max_value=20, value=10)


col1, col2, col3 = st.columns(3)

with col1 : 
    age_at_enrollment = st.number_input('Age at Enrollment', min_value=17, max_value=70, value=18)

with col2 :
    tuition_fees_up_to_date = st.selectbox('Tuition Fees Up to Date', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

with col3 : 
    scholarship_holder = st.selectbox('Scholarship Holder', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')


col1, col2, col3 = st.columns(3)

with col1 : 
    gender = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')

with col2 : 
    application_mode = st.number_input('Application Mode', min_value=1, max_value=15)

with col3 : 
    debtor = st.selectbox('Debtor', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')


# Memproses input ke dalam format array
input_data = np.array([application_mode, debtor, tuition_fees_up_to_date, gender, scholarship_holder, 
                        age_at_enrollment, curricular_units_1st_sem_approved, curricular_units_1st_sem_grade, 
                        curricular_units_2nd_sem_approved, curricular_units_2nd_sem_grade]).reshape(1, -1)

# Tombol untuk melakukan prediksi
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f"Predicted Status: **{prediction_labels.get(prediction[0], 'Unknown')}**")
