from flask import Flask, request, jsonify
import pickle
import os
import pandas as pd
from datetime import datetime
import traceback

app = Flask(__name__)

# Load your models
model_paths = {
    'arima_model': os.path.join(os.path.dirname(__file__), 'arima_model.pkl'),
    'logistic_regression_model': os.path.join(os.path.dirname(__file__), 'logistic_regression_model.pkl'),
    'program_encoder': os.path.join(os.path.dirname(__file__), 'program_encoder.pkl')
}

models = {}
for name, path in model_paths.items():
    with open(path, 'rb') as f:
        models[name] = pickle.load(f)

expected_columns = [
    'Year', 'Offense', 'Program_BAEL', 'Program_BASS', 'Program_BEED', 
    'Program_BPED', 'Program_BS PSYCH', 'Program_BSAM', 'Program_BSCE', 
    'Program_BSED', 'Program_BSHM', 'Program_BSIS', 'Program_BSIT', 
    'Program_BSPSYCH', 'Program_BTLED'
]

@app.route('/')
def home():
    return 'Server API is running'

@app.route('/predict/arima', methods=['POST'])
def predict_arima():
    data = request.json
    try:
        # Log incoming data
        print("Received data:", data)

        # Extract start date and steps from input data
        start_date_str = data.get('start_date')
        steps = data.get('steps', 4)
        if not start_date_str:
            return jsonify({'error': 'Start date is required'}), 400

        # Parse start date
        start_date = pd.to_datetime(start_date_str, dayfirst=True)
        print("Parsed start date:", start_date)

        # Check model structure and latest date
        arima_model = models.get('arima_model')
        if arima_model is None:
            return jsonify({'error': 'ARIMA model not found'}), 500
        print("ARIMA model loaded successfully.")

        
        #convert to numpy arima_model, add seperate var that is converted
        if isinstance(arima_model.data.endog, pd.Series):
            arima_model_numpy = arima_model.data.endog.to_numpy()
        else:
            arima_model_numpy = arima_model.data.endog
        
        print(f"The type is: {type(arima_model_numpy)}")

        # Ensure the model's data end date is accessible
        last_date = arima_model.data.endog.index[-1] if hasattr(arima_model.data.endog, 'index') else None
        if last_date is not None and not isinstance(last_date, pd.Timestamp):
            last_date = pd.to_datetime(last_date, dayfirst=True)

        # Calculate weeks to forecast
        weeks_ahead = (start_date - last_date).days // 7
        if weeks_ahead < 0:
            return jsonify({'error': 'Start date must be after the last date in the dataset'}), 400
        print("Weeks ahead:", weeks_ahead)

        # Total forecast steps needed
        total_steps = weeks_ahead + steps
        print("Total steps for forecast:", total_steps)

        # Generate the forecast
        forecast = arima_model.forecast(steps=total_steps)
        print("Forecast generated:", forecast)

        # Filter forecast to start date and onward
        relevant_forecast = forecast[weeks_ahead:]
        forecast_rounded = relevant_forecast.round().tolist()
        print("Rounded forecast:", forecast_rounded)

        # Create response with dates formatted as dd/mm/yyyy
        future_dates = pd.date_range(start=start_date, periods=steps, freq='W')
        response = [{'date': date.strftime('%d/%m/%Y'), 'prediction': pred} for date, pred in zip(future_dates, forecast_rounded)]
        print("Response:", response)

        return jsonify({'prediction': response})

    except Exception as e:
        error_message = traceback.format_exc()
        print("Error occurred:\n", error_message)
        return jsonify({'error': str(e)}), 500

#add prints for debug
@app.route('/predict/logistic', methods=['POST'])
def predict_logistic():
    # Step 1: Extract data from JSON request
    data = request.json

    # Step 2: Create DataFrame from input data
    new_data = pd.DataFrame({
        'Program': [data['Program']], 
        'Year': [data['Year_level']],       
        'Offense': [data['Offense']]
    })

    # new_data = pd.DataFrame({
    # 'Program': ['BSIT'], 
    # 'Year_Level': [4],       
    # 'Offense': [1]           
    # })

    # Step 3: Encode categorical feature(s) using the encoder from models
    encoder = models['program_encoder']
    if hasattr(encoder, 'categories_'):
        print("good to g")
    else:
        print("not fitted")
    encoded_new_data = encoder.transform(new_data[['Program']]).toarray()
    encoded_new_data_df = pd.DataFrame(encoded_new_data, columns=encoder.get_feature_names_out(['Program']))

    # Step 4: Combine encoded data with the remaining features
    new_data_prepared = pd.concat([new_data.drop(['Program'], axis=1), encoded_new_data_df], axis=1)

    # Step 5: Align columns with expected structure
    new_data_prepared = new_data_prepared.reindex(columns=expected_columns, fill_value=0)

    # Step 6: Predict re-offending probability and class
    model = models['logistic_regression_model']
    probability = model.predict_proba(new_data_prepared)[0][1]
    prediction = model.predict(new_data_prepared)[0]

    # Step 7: Return the prediction and probability as JSON response
    return jsonify({
        'prediction': int(prediction),  # Convert to int for JSON serialization
        'probability': round(probability, 2)  # Round probability for readability
    })

@app.route('/predict/encoder', methods=['POST'])
def predict_encoder():
    data = request.json

    prediction = models['program_encoder'].predict([data['input']]).tolist()
    return jsonify({'prediction': prediction})

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, Nice your API is now WORKING!'})

if __name__ == '__main__':
    app.run(port=5000)
