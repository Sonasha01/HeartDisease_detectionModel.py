# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 20:31:54 2024

@author: SOSA
"""
import numpy as np
import pickle



#loading the saved model
loaded_model = pickle.load(open('C:/Users/SOSA/OneDrive/clgggg/Desktop/Heart Disease Prediction/trained_model.sav1', 'rb'))


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