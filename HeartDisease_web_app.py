# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 20:35:53 2024

@author: SOSA
"""
import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('C:/Users/SOSA/OneDrive/clgggg/Desktop/Heart Disease Prediction/trained_model.sav1', 'rb'))


#creating a function for prediction

def HeartDisease_prediction(input_data):
   
    input_data = (52,1,0,125,212,0,1,168,0,1,2,2,3)  

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      print('The person does not have any heart disease')
    else:
      print('The person does have heart disease')


def main():
    
    
    #giving a title
    st.title ('HeartDisease Prediction Web Page')

#getting title the input from the user
	
age = st.text_input('Age of the person')
sex = st.text_input('Sex of the person')
cp = st.text_input('cp value')
trestbps = st.text_input('trestbp value')
chol = st.text_input('chol value')
fbs = st.text_input('fbs value')
restecg = st.text_input('restecg value')
thalach = st.text_input('thalach value')
exang = st.text_input('exang value')
oldpeak = st.text_input('oldpeak value')
slope = st.text_input('slope value')
ca = st.text_input('ca value')
thal = st.text_input('thal value')




#code for prediction
diagnosis = ' '
# creating a button for Prediction 
if st.button('HeartDisease Test Result'):
 diagnosis = HeartDisease_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
st.success(diagnosis)

if  __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
