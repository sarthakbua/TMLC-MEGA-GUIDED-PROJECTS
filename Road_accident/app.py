
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from prediction import get_prediction, ordinal_encoder

model = joblib.load(r'EDA/RTA/Model/xtr_final.joblib')

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

options_light_condition = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
       'Darkness - lights unlit']

options_vehicle_owner = ['Owner', 'Governmental', 'Organization', 'Other']

options_num_vehicles = [2, 1, 3, 6, 4, 7]

options_num_of_casualties = [2, 1, 3, 4, 6, 5, 8, 7]

options_casualty_class = ['na', 'Driver or rider', 'Pedestrian', 'Passenger']

options_defect_vehicle = ['No defect', '7', '5']

options_pedestrian= ['Not a Pedestrian', "Crossing from driver's nearside",
       'Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle',
       'Unknown or other',
       'Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle',
       'In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)',
       'Walking along in carriageway, back to traffic',
       'Walking along in carriageway, facing traffic',
       'In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle']

options_sex_of_driver = ['Male', 'Female', 'Unknown']

options_road_condition = ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep']

options_service_vehicle = ['Above 10yr', '5-10yrs', '1-2yr', '2-5yrs', 'Unknown',
                            'Below 1yr']

options_casualty_fitness = ['Normal', 'Deaf', 'Other', 'Blind', 'NormalNormal']

options_vehicle_owner_relation = ['Employee', 'Unknown', 'Owner', 'Other']


features_new = ['Type_of_vehicle', 'Owner_of_vehicle']

st.markdown("<h1 style = 'text-align: center;'>Accident Severity Prediction </h1>", unsafe_allow_html = True)

def main():
    with st.form('prediction_form'):

        st.subheader("Input following features")
        defect_of_vehicle = st.selectbox("Select Vehicle Defect: ", options = options_defect_vehicle)
        vehicle_type = st.selectbox("Select Vehicle Type: ", options = options_vehicle_type)
        owner_of_vehicle= st.selectbox("Select Vehicle Owner: ", options = options_vehicle_owner)
        road_surface = st.selectbox("Select Road Surface condition: ", options = options_road_condition)
        driver_age = st.selectbox("Select Driver Age: ", options = options_age)
        driving_experience = st.selectbox("Select Driving Experience: ", options = options_driver_exp)
        light_conditions = st.selectbox("select light conditions:", options = options_light_condition)
        service_years = st.selectbox("select service years:", options = options_service_vehicle)


        submit = st.form_submit_button("Predict")

    if submit:
        defect_of_vehicle = ordinal_encoder(defect_of_vehicle, options_defect_vehicle)
        vehicle_type = ordinal_encoder(vehicle_type, options_vehicle_type)
        owner_of_vehicle = ordinal_encoder(owner_of_vehicle, options_vehicle_owner)
        road_surface = ordinal_encoder(road_surface, options_road_condition)
        driver_age = ordinal_encoder(driver_age, options_age)
        driving_experience = ordinal_encoder(driving_experience, options_driver_exp)
        light_conditions = ordinal_encoder(light_conditions, options_light_condition)
        service_years = ordinal_encoder(service_years, options_service_vehicle)

        data = np.array([defect_of_vehicle, vehicle_type, light_conditions, owner_of_vehicle,
                           driving_experience, road_surface, driver_age, service_years]).reshape(1,-1)

        pred = get_prediction(data=data, model = model)

        st.write(f"the predicted Severity is: {pred[0]}")

if __name__ == '__main__':
    main()        