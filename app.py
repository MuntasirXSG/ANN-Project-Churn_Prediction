
import streamlit as st
import tensorflow as tf

import pickle
import pandas as pd
import numpy as np


model = tf.keras.models.load_model('model.h5')

with open('encoded_country.pkl','rb') as file:
    encoded_country=pickle.load(file)
with open('encoded_gender.pkl','rb') as file :
    encoded_gender= pickle.load(file)
with open('Scaler.plk','rb') as file :
      scaler= pickle.load(file)    



## streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', encoded_country.categories_[0])
gender = st.selectbox('Gender', encoded_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [encoded_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


eg = encoded_country.transform([[geography]]).toarray()
geo_df = pd.DataFrame(eg, columns=encoded_country.get_feature_names_out(['Geography']))




input_data = pd.concat([input_data, geo_df], axis =1)
scaled_input = scaler.transform(input_data)
prediction = model.predict(scaled_input)


proba = prediction [0][0]

st.write(f'churn probability: {proba}')
