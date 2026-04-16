import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Define the features for preprocessing (must match training)
env_features_app = ['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
                    'ClimateChange', 'Siltation', 'Watersheds', 'WetlandLoss', 'Landslides']
urban_features_app = ['Urbanization', 'DamsQuality', 'AgriculturalPractices', 'Encroachments',
                     'IneffectiveDisasterPreparedness', 'DrainageSystems', 'CoastalVulnerability',
                     'DeterioratingInfrastructure', 'PopulationScore', 'InadequatePlanning', 'PoliticalFactors']
engineered_features_app = ['EnvironmentalIndex', 'UrbanIndex', 'Env_Urban_Interaction',
                           'EnvironmentalIndex_sq', 'UrbanIndex_sq', 'Env_Urban_Interaction_sq']

st.title("Flood Risk Prediction App")

st.write("Enter the values for the environmental and urban indicators to predict flood probability.")

# Input fields for all original features
input_data = {}
st.header("Environmental Indicators")
for feature in env_features_app:
    input_data[feature] = st.slider(f"{feature} (1-10)", 1, 10, 5)
st.header("Urban Indicators")
for feature in urban_features_app:
    input_data[feature] = st.slider(f"{feature} (1-10)", 1, 10, 5)

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Feature Engineering for input data
input_df['EnvironmentalIndex'] = input_df[env_features_app].mean(axis=1)
input_df['UrbanIndex'] = input_df[urban_features_app].mean(axis=1)
input_df['Env_Urban_Interaction'] = input_df['EnvironmentalIndex'] * input_df['UrbanIndex']
input_df['EnvironmentalIndex_sq'] = input_df['EnvironmentalIndex']**2
input_df['UrbanIndex_sq'] = input_df['UrbanIndex']**2
input_df['Env_Urban_Interaction_sq'] = input_df['Env_Urban_Interaction']**2

# Select and scale engineered features
input_scaled = scaler.transform(input_df[engineered_features_app])

# Prediction
if st.button("Predict Flood Probability"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Flood Probability: {prediction:.4f}")
