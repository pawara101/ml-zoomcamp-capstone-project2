import streamlit as st
import torch
from train import AqiCNN,city_label_encoder

model = AqiCNN()
model.load_state_dict(torch.load('models/model.pth'))
model.eval()

test_input = {
    'city': 31,
    'co': 0.5,
    'o3': 0.03,
    'no2': 12.3,
    'so2': 4.5,
    'pm10': 45.0,
    'pm25': 22.0
}

city_list = ['Montgomery', 'Juneau', 'Phoenix', 'Little Rock', 'Sacramento',
       'Denver', 'Hartford', 'Dover', 'Tallahassee', 'Atlanta',
       'Honolulu', 'Boise', 'Springfield', 'Indianapolis', 'Des Moines',
       'Topeka', 'Frankfort', 'Baton Rouge', 'Augusta', 'Annapolis',
       'Boston', 'Lansing', 'Saint Paul', 'Jackson', 'Jefferson City',
       'Helena', 'Lincoln', 'Carson City', 'Concord', 'Trenton',
       'Santa Fe', 'Albany', 'Raleigh', 'Bismarck', 'Columbus',
       'Oklahoma City', 'Salem', 'Harrisburg', 'Providence', 'Columbia',
       'Pierre', 'Nashville', 'Austin', 'Salt Lake City', 'Montpelier',
       'Richmond', 'Olympia', 'Charleston', 'Madison', 'Cheyenne',
       'Washington']

st.title("Predict AQI Score")
st.write("Input the pollutant levels to predict the AQI score.")

city = st.selectbox('City', options=city_list, index=test_input['city'])
city_encoded = city_label_encoder.transform([city])[0]

co = st.number_input('CO level', value=test_input['co'])
o3 = st.number_input('O3 level', value=test_input['o3'])
no2 = st.number_input('NO2 level', value=test_input['no2'])
so2 = st.number_input('SO2 level', value=test_input['so2'])
pm10 = st.number_input('PM10 level', value=test_input['pm10'])
pm25 = st.number_input('PM2.5 level', value=test_input['pm25'])

input_tensor = torch.tensor(
    [[city_encoded, co, o3, no2, so2, pm10, pm25]],
    dtype=torch.float32
)

with torch.no_grad():
    prediction = model(input_tensor)
    aqi_score = prediction.item()

st.subheader("Predicted AQI Score")
st.write(f"The predicted AQI score is: {aqi_score:.2f}")