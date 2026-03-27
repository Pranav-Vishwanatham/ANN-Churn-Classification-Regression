import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained models
model_churn = tf.keras.models.load_model('churn_classification_model.h5')
model_salary = tf.keras.models.load_model('regression_model.h5')

# Load the encoders and scalers
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler_classification.pkl', 'rb') as file:
    scaler_churn = pickle.load(file)

with open('scaler_regression.pkl', 'rb') as file:
    scaler_salary = pickle.load(file)

## streamlit app
st.title('Customer Churn and Salary Prediction')

# Create tabs
tab1, tab2 = st.tabs(["Churn Prediction", "Salary Prediction"])

with tab1:
    st.header("Churn Prediction")
    
    # User input
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0], key='geo_churn')
    gender = st.selectbox('Gender', label_encoder_gender.classes_, key='gender_churn')
    age = st.slider('Age', 18, 92, key='age_churn')
    balance = st.number_input('Balance', key='balance_churn')
    credit_score = st.number_input('Credit Score', key='credit_churn')
    estimated_salary = st.number_input('Estimated Salary', key='salary_churn')
    tenure = st.slider('Tenure', 0, 10, key='tenure_churn')
    num_of_products = st.slider('Number of Products', 1, 4, key='products_churn')
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], key='card_churn')
    is_active_member = st.selectbox('Is Active Member', [0, 1], key='active_churn')

    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler_churn.transform(input_data)

    # Predict churn
    prediction = model_churn.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.write(f'Churn Probability: {prediction_proba:.2f}')

    if prediction_proba > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')

with tab2:
    st.header("Salary Prediction")
    
    # User input (same as churn but no estimated salary)
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0], key='geo_salary')
    gender = st.selectbox('Gender', label_encoder_gender.classes_, key='gender_salary')
    age = st.slider('Age', 18, 92, key='age_salary')
    balance = st.number_input('Balance', key='balance_salary')
    credit_score = st.number_input('Credit Score', key='credit_salary')
    tenure = st.slider('Tenure', 0, 10, key='tenure_salary')
    num_of_products = st.slider('Number of Products', 1, 4, key='products_salary')
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], key='card_salary')
    is_active_member = st.selectbox('Is Active Member', [0, 1], key='active_salary')

    # Let user choose Exited for salary prediction
    exited = st.selectbox('Exited', [0, 1], key='exited_salary')

    # Prepare input for salary prediction (including Exited)
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'Exited': [exited]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler_salary.transform(input_data)

    # Predict salary
    prediction = model_salary.predict(input_data_scaled)
    predicted_salary = prediction[0][0]

    st.write(f'Predicted Estimated Salary: ${predicted_salary:,.2f}')

