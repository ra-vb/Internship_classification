from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load the trained model pipeline
model = joblib.load('logistic_model.pkl')

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        data = request.get_json()

        # Convert to DataFrame for preprocessing
        input_df = pd.DataFrame([data])

        # Predict satisfaction score
        prediction = model.predict(input_df)[0]

        # Map result to label
        result = "High satisfaction score" if prediction == 1 else "Low satisfaction score"

        return jsonify({
            "input": data,
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
