
import streamlit as st
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Flood Risk Predictor",
    page_icon="🌊",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# ---------------- FEATURES ----------------
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

# ---------------- HEADER ----------------
st.title("🌊 Flood Risk Prediction Dashboard")
st.markdown("### Smart Flood Risk Assessment using Machine Learning")

st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Input Parameters")
st.sidebar.markdown("Adjust indicators (1 = Low, 10 = High)")

input_data = {}

st.sidebar.subheader("🌱 Environmental")
for f in env_features:
    input_data[f] = st.sidebar.slider(f, 1, 10, 5)

st.sidebar.subheader("🏙️ Urban")
for f in urban_features:
    input_data[f] = st.sidebar.slider(f, 1, 10, 5)

# ---------------- DATA PROCESSING ----------------
df = pd.DataFrame([input_data])

df['EnvironmentalIndex'] = df[env_features].mean(axis=1)
df['UrbanIndex'] = df[urban_features].mean(axis=1)
df['Env_Urban_Interaction'] = df['EnvironmentalIndex'] * df['UrbanIndex']

df['EnvironmentalIndex_sq'] = df['EnvironmentalIndex']**2
df['UrbanIndex_sq'] = df['UrbanIndex']**2
df['Env_Urban_Interaction_sq'] = df['Env_Urban_Interaction']**2

X = scaler.transform(df[features])

# ---------------- MAIN DISPLAY ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Prediction Output")

    if st.button("🚀 Predict Flood Risk"):
        pred = model.predict(X)[0]

        st.progress(float(pred))
        st.metric(label="Flood Risk Score", value=f"{pred:.2f}")

        if pred < 0.4:
            st.success("✅ Low Risk")
        elif pred < 0.7:
            st.warning("⚠️ Moderate Risk")
        else:
            st.error("🚨 High Risk")

with col2:
    st.subheader("📌 Insights")
    st.info(f"Environmental Index: {df['EnvironmentalIndex'].values[0]:.2f}")
    st.info(f"Urban Index: {df['UrbanIndex'].values[0]:.2f}")

st.markdown("---")

with st.expander("ℹ️ About"):
    st.write("This app predicts flood risk using a trained ML model.")
