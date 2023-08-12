import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Creating a function fro prediction

def diabetes_prediction(input_data):

    # Changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshaping the input data as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return "The female is Non-diabetic."
    else:
        return "The female is diabetic."


def main():


    # Giving a title
    st.title("Diabetes Prediction Web App")

    # Getting the input data from the user
    
    Pregnancies = st.text_input("Number of Pregnancies:")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("Skin Thickness Value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Age = st.text_input("Age of the Person")

    # Code for Prediction
    diagnosis = ""

    # Creating a button for Prediction

    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)


if __name__ == "__main__":
    main()