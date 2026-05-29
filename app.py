import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Load
model = joblib.load("Models/model.pkl")
encoders = joblib.load("Models/label_encoders.pkl")

# Page
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌍",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.stButton>button{
background:#00ADB5;
color:white;
border-radius:10px;
width:100%;
height:3em;
font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🌍 AQI Prediction Dashboard")
st.write("Random Forest Regression")

# Sidebar
st.sidebar.header("Enter Pollution Details")

country = st.sidebar.selectbox(
    "Country",
    encoders['Country'].classes_
)

city = st.sidebar.selectbox(
    "City",
    encoders['City'].classes_
)

co = st.sidebar.slider(
    "CO AQI Value",
    0,500,50
)

ozone = st.sidebar.slider(
    "Ozone AQI Value",
    0,500,50
)

no2 = st.sidebar.slider(
    "NO2 AQI Value",
    0,500,50
)

pm25 = st.sidebar.slider(
    "PM2.5 AQI Value",
    0,500,50
)

co_cat = st.sidebar.selectbox(
    "CO Category",
    encoders['CO AQI Category'].classes_
)

ozone_cat = st.sidebar.selectbox(
    "Ozone Category",
    encoders['Ozone AQI Category'].classes_
)

no2_cat = st.sidebar.selectbox(
    "NO2 Category",
    encoders['NO2 AQI Category'].classes_
)

pm25_cat = st.sidebar.selectbox(
    "PM2.5 Category",
    encoders['PM2.5 AQI Category'].classes_
)

# Encode
data = pd.DataFrame([{

    'Country':
    encoders['Country'].transform([country])[0],

    'City':
    encoders['City'].transform([city])[0],

    'CO AQI Value':co,

    'CO AQI Category':
    encoders['CO AQI Category'].transform([co_cat])[0],

    'Ozone AQI Value':ozone,

    'Ozone AQI Category':
    encoders['Ozone AQI Category'].transform([ozone_cat])[0],

    'NO2 AQI Value':no2,

    'NO2 AQI Category':
    encoders['NO2 AQI Category'].transform([no2_cat])[0],

    'PM2.5 AQI Value':pm25,

    'PM2.5 AQI Category':
    encoders['PM2.5 AQI Category'].transform([pm25_cat])[0]
}])

# Predict
if st.button("Predict AQI"):

    pred = model.predict(data)[0]

    st.success(
        f"Predicted AQI: {pred:.2f}"
    )

    # AQI Gauge
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pred,
            title={'text':"AQI"},
            gauge={
                'axis':{'range':[0,500]}
            }
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Bar Graph
    graph_df = pd.DataFrame({

        'Pollutant':
        ['CO','Ozone','NO2','PM2.5'],

        'Value':
        [co, ozone, no2, pm25]
    })

    bar = px.bar(
        graph_df,
        x='Pollutant',
        y='Value',
        title='Pollution Levels'
    )

    st.plotly_chart(
        bar,
        use_container_width=True
    )

    # Radar Chart
    radar = go.Figure()

    radar.add_trace(
        go.Scatterpolar(
            r=[co, ozone, no2, pm25],
            theta=[
                'CO',
                'Ozone',
                'NO2',
                'PM2.5'
            ],
            fill='toself'
        )
    )

    radar.update_layout(
        showlegend=False,
        title="Pollutant Radar"
    )

    st.plotly_chart(
        radar,
        use_container_width=True
    )
