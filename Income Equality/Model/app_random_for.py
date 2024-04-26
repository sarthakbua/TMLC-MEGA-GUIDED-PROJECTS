# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:02:22 2024

@author: Sarthak
"""

import streamlit as st
import pandas as pd
import numpy as np 
import joblib
 
from sklearn.ensemble import RandomForestClassifier
from  prediction import get_prediction, ordinal_encoder

model = joblib.load(r'Random_Forest_Classifier.joblib')

st.set_page_config(page_title="Income unequality Prediction", layout="wide")


gender_male = ['M', 'F']
tax_status_filer = [0,1]


st.markdown("<h1 style = 'text-align: center;'>Income Inequality </h1>", unsafe_allow_html = True)





def main():
    with st.form('prediction_form'):

        st.subheader("Input following features")
        
        
        occupation_code = st.slider("Occupation Code: ", 0, 80, value=0, format="%d")
        age = st.slider("Age: ", 1, 90, value=0, format="%d")  
        working_week_per_year = st.slider("Working week per year: ", 1, 90, value=0, format="%d")  
        total_employed= st.slider("Total Employed: ", 1, 90, value=0, format="%d") 
        stock_status = st.number_input('Insert stock status')
        st.write('Input value ', stock_status)
        industrycode = st.slider("Industry Code: ", 1, 90, value=0, format="%d")  
        gender=st.selectbox ("Select Gender: ", options = gender_male )
        gains = st.number_input('Gain')
        st.write('Input value ', gains)
        tax_status_nfiler=st.selectbox ("Select Tax input: ", options = tax_status_filer )
        
        submit = st.form_submit_button("Predict")
        

    if submit:
        gender = ordinal_encoder(gender, gender_male)


        data = np.array(['occupation_code', 'age', 'working_week_per_year', 'total_employed', 'vehicles_involved',
                          'stock_status', 'industrycode', 'gender', 'gains','tax_status_nfiler']).reshape(1,-1)

        
        pred = get_prediction(data=data, model = model)

        st.write(f"the predicted inequality is: {pred[0]}")       
        
if __name__ == '__main__':
    main()       