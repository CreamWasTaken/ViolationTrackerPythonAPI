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
    try:
        # Get the number of weeks to forecast from request
        data = request.json
        steps = data.get('steps', 12)  # Default to 12 weeks if not provided

        # Generate predictions
        arima_model = models['arima_model']
        predictions = arima_model.forecast(steps=steps)
        predictions_rounded = predictions.round().tolist()  # Round and convert to list for JSON response

        # Create date range for the predictions
        last_date = data.get('last_date', pd.Timestamp.now())
        future_dates = pd.date_range(start=last_date, periods=steps, freq='W').strftime('%Y-%m-%d').tolist()

        # Build response
        response = {
            "predictions": {
                "dates": future_dates,
                "values": predictions_rounded
            }
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
