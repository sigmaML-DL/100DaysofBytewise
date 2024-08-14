import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model and preprocessor
with open('C:/Users/abdul/OneDrive/Desktop/ByteWise_ML/Projects/Flight Delay Prediction/Delay.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('C:/Users/abdul/OneDrive/Desktop/ByteWise_ML/Projects/Flight Delay Prediction/Pre-process.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

categorical_columns = ['Airlines', 'OriginCityName', 'DestCityName']
numeric_columns = ['Month', 'DayOfWeek', 'DepDelay', 'AirTime', 'Distance']

airlines_list = ['American Airlines', 'Delta Airlines', 'Southwest Airlines', 'United Airlines']
origin_cities_list = ['Chicago, IL', 'Atlanta, GA', 'New York, NY', 'Denver, CO', 'Dallas/Fort Worth, TX']
dest_cities_list = ['Chicago, IL', 'Atlanta, GA', 'New York, NY', 'Denver, CO', 'Dallas/Fort Worth, TX']
months_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']  
days_of_week_list = ['1', '2', '3', '4', '5', '6', '7'] 

st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4CAF50; 
        color: white; 
        padding: 10px 20px; 
        border: none; 
        border-radius: 5px; 
        font-size: 16px;
    }
    .sidebar .sidebar-content {
        position: relative;
        height: 100%;
    }
    .sidebar .sidebar-content .stButton {
        position: absolute;
        bottom: 0;
        width: 100%;
    }
    .center-text {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Flight Delay Prediction')

# Define sidebar inputs
with st.sidebar:
    st.header('Input Parameters')

    airlines = st.selectbox('Airlines', options=airlines_list)
    origin_city = st.selectbox('Origin City', options=origin_cities_list)
    dest_city = st.selectbox('Destination City', options=dest_cities_list)
    month = st.selectbox('Month', options=months_list)
    day_of_week = st.selectbox('Day of Week', options=days_of_week_list)

    dep_delay = st.slider('Departure Delay (in minutes)', min_value=0, max_value=180, value=0)
    air_time = st.slider('Air Time (in minutes)', min_value=0, max_value=600, value=0)
    distance = st.slider('Distance (in miles)', min_value=0, max_value=3000, value=0)

    st.sidebar.write("")  # Add an empty line for spacing
    st.sidebar.write("")  # Adjust this as needed for more space

    predict_button = st.sidebar.button('Predict')

# Create input DataFrame and make prediction when button is clicked
if predict_button:
    X_input = pd.DataFrame({
        'Airlines': [airlines],
        'OriginCityName': [origin_city],
        'DestCityName': [dest_city],
        'Month': [month],
        'DayOfWeek': [day_of_week],
        'DepDelay': [dep_delay],
        'AirTime': [air_time],
        'Distance': [distance]
    })

    X_input_preprocessed = preprocessor.transform(X_input)
    prediction = model.predict(X_input_preprocessed)[0]
    st.markdown(f"<h2 class='center-text'><strong>{'Your Flight is likely to be Delayed' if prediction == 1 else 'Your Flight is On Time'}</strong></h2>", unsafe_allow_html=True)

