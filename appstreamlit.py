import streamlit as st
import pandas as pd
import joblib
#import pickle

# Load the trained model
model = joblib.load('iris_model.pkl')

# Define the app
st.title("Iris Flower Classifier")
st.write("Enter the flower measurements below to predict the species:")

# Input fields for user data
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.slider("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    # Make prediction
    prediction = model.predict(input_data)
    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"The predicted species is: {species[prediction[0]]}")