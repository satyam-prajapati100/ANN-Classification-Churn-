import streamlit as  st 
import numpy as np 
import tensorflow as tf 
from sklearn.preprocessing import  StandardScaler, LabelEncoder , OneHotEncoder
import pandas as pd
import pickle


# load the trained model 
model = tf.keras.models.load_model('model.h5') 

# load encoders and scaler 

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)   

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)     

with open('scaler.pkl', 'rb') as f:     
    scaler = pickle.load(f) 


#  streamlit app 

st.title("Customer Churn Prediction") 

#  user input for the features 

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", value=0.0)
credit_score = st.number_input("Credit Score", value=0.0)
estimated_salary = st.number_input("Estimated Salary", value=0.0)
tenure = st.slider("Tenure", min_value=0, max_value=10, value=0)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# create a dataframe from the user input 

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


# one hot encode the geography column 
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


# combine one hot encoded columns with the input data
input_data = pd.concat([input_data.reset_index(drop=True) , geo_encoded_df], axis=1)

#  Scale the input data 
input_data_scaled = scaler.transform(input_data) 

# predict the churn probability 
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f"Churn Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")
