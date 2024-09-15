from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load models and preprocessor
cnn_lstm_model = pickle.load(open('cnn_lstm_model.pkl', 'rb'))
rnn_model = pickle.load(open('rnn_model.pkl', 'rb'))
dtr_model = pickle.load(open('dtr_model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Get data from request
            Year = float(request.form['Year'])
            average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
            pesticides_tonnes = float(request.form['pesticides_tonnes'])
            avg_temp = float(request.form['avg_temp'])
            Area = request.form['Area']
            Item = request.form['Item']
            
            # Create feature array
            features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
            
            # Transform features
            transformed_features = preprocessor.transform(features)
            
            # Reshape for CNN-LSTM
            features_reshaped = np.reshape(transformed_features, (transformed_features.shape[0], transformed_features.shape[1], 1))
            
            # Predictions
            cnn_lstm_prediction = cnn_lstm_model.predict(features_reshaped)
            rnn_prediction = rnn_model.predict(features_reshaped)
            dtr_prediction = dtr_model.predict(transformed_features)
            
            # Create prediction dictionary
            prediction = {
                'CNN-LSTM Prediction': cnn_lstm_prediction[0],
                'RNN Prediction': rnn_prediction[0],
                'Predicting': dtr_prediction[0]
            }
        except Exception as e:
            prediction = {'error': str(e)}
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
