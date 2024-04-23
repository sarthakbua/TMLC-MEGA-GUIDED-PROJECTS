
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from prediction import get_prediction, ordinal_encoder

model = joblib.load(r'EDA/RTA/Model/xgb4.joblib')

st.set_page_config(page_title="Accident Severity Prediction", layout="wide")

# Creating Option Dropdown

options_day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
options_age = ['18-30', '31-50', 'Under 18', 'Over 51', 'Unknown']

options_acc_area = ['Residential areas', 'Office areas', '  Recreational areas',
       ' Industrial areas', 'Other', ' Church areas',
       '  Market areas', 'Unknown', 'Rural village areas',
       ' Outside rural areas', ' Hospital areas', 'School areas',
       'Rural village areasOffice areas', 'Recreational areas']

options_cause = ['Moving Backward', 'Overtaking', 'Changing lane to the left',
       'Changing lane to the right', 'Overloading', 'Other',
       'No priority to vehicle', 'No priority to pedestrian',
       'No distancing', 'Getting off the vehicle improperly',
       'Improper parking', 'Overspeed', 'Driving carelessly',
       'Driving at high speed', 'Driving to the left', 'Unknown',
       'Overturning', 'Turnover', 'Driving under the influence of drugs',
       'Drunk driving']

options_vehicle_type = ['Automobile', 'Public (> 45 seats)', 'Lorry (41?100Q)',
       'Public (13?45 seats)', 'Lorry (11?40Q)', 'Long lorry',
       'Public (12 seats)', 'Taxi', 'Pick up upto 10Q', 'Stationwagen',
       'Ridden horse', 'Other', 'Bajaj', 'Turbo', 'Motorcycle',
       'Special vehicle', 'Bicycle']

options_driver_exp = ['1-2yr', 'Above 10yr', '5-10yr', '2-5yr', 'No Licence',
       'Below 1yr', 'unknown']

options_lanes = ['Undivided Two way', 'other', 'Double carriageway (median)',
       'One way', 'Two-way (divided with solid lines road marking)',
       'Two-way (divided with broken lines road marking)', 'Unknown']

features = ['hour', 'day_of_week', 'casualties', 'accident_cause', 'vehicles_involved', 'vehicle_type', 'driver_age', 
            'accident_area', 'driving_experience', 'lanes']

st.markdown("<h1 style = 'text-align: center;'>Accident Severity Prediction </h1>", unsafe_allow_html = True)

def main():
    with st.form('prediction_form'):

        st.subheader("Input following features")

        hour = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
        day_of_week = st.selectbox("Select Day of Week: ", options = options_day)
        casualties = st.slider("Hour of Accident: ", 1, 8, value=0, format="%d")
        accident_cause = st.selectbox("Select Accident Cause: ", options = options_cause)
        vehicles_involved = st.slider("Pickup Hour: ", 1, 7, value = 0, format = "%d")
        vehicle_type = st.selectbox("Select Vehicle Type: ", options = options_vehicle_type)
        driver_age = st.selectbox("Select Driver Age: ", options = options_age)
        accident_area = st.selectbox("Select Accident Area: ", options = options_acc_area)
        driving_experience = st.selectbox("Select Driving Experience: ", options = options_driver_exp)
        lanes = st.selectbox("Select Lanes: ", options = options_lanes)

        submit = st.form_submit_button("Predict")

    if submit:
        day_of_week = ordinal_encoder(day_of_week, options_day)
        accident_cause = ordinal_encoder(accident_cause, options_cause)
        vehicle_type = ordinal_encoder(vehicle_type, options_vehicle_type)
        driver_age = ordinal_encoder(driver_age, options_age)
        accident_area = ordinal_encoder(accident_area, options_acc_area)
        driving_experience = ordinal_encoder(driving_experience, options_driver_exp)
        lanes = ordinal_encoder(lanes, options_lanes)

        data = np.array(['hour', 'day_of_week', 'casualties', 'accident_cause', 'vehicles_involved',
                          'vehicle_type', 'driver_age', 'accident_area', 'driving_experience', 'lanes']).reshape(1,-1)

        pred = get_prediction(data=data, model = model)

        st.write(f"the predicted Severity is: {pred[0]}")

if __name__ == '__main__':
    main()        