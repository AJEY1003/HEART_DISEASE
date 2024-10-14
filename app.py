from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the K-NN model (assuming it's saved in the same directory)
model = joblib.load(r"C:\Users\AJEY K\Desktop\AI-1\knn_model.pkl")


# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')  # Create a simple homepage

# Define a route for predictions (POST method)
@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form in the frontend
    int_features = [float(x) for x in request.form.values()]
    
    # Convert the features into an array (reshape for single prediction)
    final_features = np.array(int_features).reshape(1, -1)
    
    # Make the prediction using the loaded model
    prediction = model.predict(final_features)
    
    # Map prediction (0 or 1) to disease presence
    output = 'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease'
    
    return render_template('index.html', prediction_text=f'Result: {output}')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=False)



