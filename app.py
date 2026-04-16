import streamlit as st
import pickle
import pandas as pd

# Load model and scaler
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# Feature groups
env_features = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
    'ClimateChange', 'Siltation', 'Watersheds', 'WetlandLoss', 'Landslides'
]

urban_features = [
    'Urbanization', 'DamsQuality', 'AgriculturalPractices', 'Encroachments',
    'IneffectiveDisasterPreparedness', 'DrainageSystems', 'CoastalVulnerability',
    'DeterioratingInfrastructure', 'PopulationScore', 'InadequatePlanning', 'PoliticalFactors'
]

features = [
    'EnvironmentalIndex', 'UrbanIndex', 'Env_Urban_Interaction',
    'EnvironmentalIndex_sq', 'UrbanIndex_sq', 'Env_Urban_Interaction_sq'
]

st.title("Flood Risk Prediction App")

# Inputs
input_data = {}

st.header("Environmental Indicators")
for f in env_features:
    input_data[f] = st.slider(f, 1, 10, 5)

st.header("Urban Indicators")
for f in urban_features:
    input_data[f] = st.slider(f, 1, 10, 5)

# Convert input
df = pd.DataFrame([input_data])

# Feature engineering
df['EnvironmentalIndex'] = df[env_features].mean(axis=1)
df['UrbanIndex'] = df[urban_features].mean(axis=1)
df['Env_Urban_Interaction'] = df['EnvironmentalIndex'] * df['UrbanIndex']

df['EnvironmentalIndex_sq'] = df['EnvironmentalIndex']**2
df['UrbanIndex_sq'] = df['UrbanIndex']**2
df['Env_Urban_Interaction_sq'] = df['Env_Urban_Interaction']**2

# Scale
X = scaler.transform(df[features])

# Predict
if st.button("Predict"):
    pred = model.predict(X)[0]
    st.success(f"Flood Risk Probability: {pred:.4f}")
