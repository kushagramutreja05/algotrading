from flask import Flask, render_template, request,jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keras.models import load_model
from flask_cors import CORS
import tensorflow as tf
import keras 
from keras.layers import Multiply, Dense, Lambda
from keras import backend as K








app = Flask(__name__)
CORS(app)

# Load the trained GRU model
gru_model_path = 'model/gru_model_with_attention.keras'

# Load the model with custom objects
gru_model = load_model(gru_model_path,custom_objects={
   
    'tf':tf
    
},safe_mode=True)

# Load the linear regression model
linear_model_path = 'linear_regression_model.pkl'
with open(linear_model_path, 'rb') as f:
    linear_model = pickle.load(f)

# Load scalers for features (X) and target (Y)
scaler_x_path = 'model/scaler_x.pkl'
with open(scaler_x_path, 'rb') as file:
    scaler_x = pickle.load(file)

scaler_y_path = 'model/scaler_y.pkl'
with open(scaler_y_path, 'rb') as file:
    scaler_y = pickle.load(file)

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
 try:
    if request.method == 'POST':
        data = request.json

        open_price = float(data['open_price'])
        close_price = float(data['close_price'])
        high = float(data['high'])
        low = float(data['low'])
        volume = float(data['volume'])
        todays_news = data['todays_news']
        tomorrows_news = data['tomorrows_news']

        # Calculate sentiment from today's and tomorrow's news
        todays_sentiment = sid.polarity_scores(todays_news)
        compound = todays_sentiment['compound']
        
        tomorrows_sentiment = sid.polarity_scores(tomorrows_news)
        compound_shifted = tomorrows_sentiment['compound']

        # Prepare the feature array
        features = np.array([[close_price, compound, compound_shifted, volume, open_price, high, low]])
        print("Features before scaling:", type(features))
        features_scaled=pd.DataFrame(features)
        # Min-max scale the features
        features_scaled = scaler_x.transform(features_scaled)  # Use transform, not fit_transform
        features_scaled = features_scaled.reshape(features_scaled.shape + (1,))
       
        # Predict next day's close price using the GRU model
        gru_prediction = gru_model.predict(features_scaled)

        # Reshape the scaled features to a 2D array for the linear model
        features_scaled = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1])
        gru_prediction=gru_prediction.reshape(gru_prediction.shape[0], gru_prediction.shape[1])
        # Combine the GRU prediction with the scaled features
        combined_features = pd.concat([features_scaled, gru_prediction], axis=1)

        # Predict using the linear regression model
        lr_prediction = linear_model.predict(combined_features)
        lr_prediction = scaler_y.inverse_transform(lr_prediction)
        

        return jsonify({'predicted_price': lr_prediction[0][0]})
 except Exception as e:
  print(f"Prediction error: {str(e)}")
  return jsonify({'predicted_price': 'Error predicting price.'}), 500

if __name__ == '__main__':
   app.run(debug=True)
