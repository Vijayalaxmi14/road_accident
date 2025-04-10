from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np
import requests

app = Flask(__name__)
model = joblib.load('litemodel.sav')  # your trained model file

# TextLocal API setup
def send_sms(api_key, numbers, sender, message):
    url = 'https://api.textlocal.in/send/'
    data = {
        'apikey': api_key,
        'numbers': numbers,
        'sender': sender,
        'message': message
    }
    response = requests.post(url, data=data)
    return response.json()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Extract form data
        features = [
            int(request.form['Did_Police_Officer_Attend']),
            float(request.form['age_of_driver']),
            int(request.form['vehicle_type']),
            float(request.form['age_of_vehicle']),
            float(request.form['engine_cc']),
            int(request.form['day']),
            int(request.form['weather']),
            int(request.form['light']),
            int(request.form['roadsc']),
            int(request.form['gender']),
            int(request.form['speedl'])
        ]

        input_df = pd.DataFrame([features], columns=[
            'Did_Police_Officer_Attend', 'Age_of_Driver', 'Vehicle_Type', 'Age_of_Vehicle',
            'Engine_Capacity_CC', 'Day_of_Week', 'Weather_Conditions', 'Light_Conditions',
            'Road_Surface_Conditions', 'Sex_of_Driver', 'Speed_limit'
        ])

        pred = model.predict(input_df)[0]
        return str(pred)  # returned directly to the HTML via AJAX

    return render_template('index.html')


@app.route('/sms/', methods=['POST'])
def sms():
    # Predict first
    features = [
        int(request.form['Did_Police_Officer_Attend']),
        float(request.form['age_of_driver']),
        int(request.form['vehicle_type']),
        float(request.form['age_of_vehicle']),
        float(request.form['engine_cc']),
        int(request.form['day']),
        int(request.form['weather']),
        int(request.form['light']),
        int(request.form['roadsc']),
        int(request.form['gender']),
        int(request.form['speedl'])
    ]

    input_df = pd.DataFrame([features], columns=[
        'Did_Police_Officer_Attend', 'Age_of_Driver', 'Vehicle_Type', 'Age_of_Vehicle',
        'Engine_Capacity_CC', 'Day_of_Week', 'Weather_Conditions', 'Light_Conditions',
        'Road_Surface_Conditions', 'Sex_of_Driver', 'Speed_limit'
    ])

    pred = model.predict(input_df)[0]

    # Send SMS
    if pred == 1:  # FATAL
        response = send_sms(
            api_key='your_textlocal_api_key',
            numbers='91xxxxxxxxxx',  # Replace with verified number
            sender='TXTLCL',
            message='⚠️ FATAL accident risk detected! Take precaution.'
        )
        print(response)
        return "SMS Sent: FATAL ACCIDENT RISK"
    else:
        return f"No SMS needed. Predicted Severity: {pred}"


if __name__ == '__main__':
    app.run(debug=True)
