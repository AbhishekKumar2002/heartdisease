from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Load the trained model
heart_model = joblib.load("heart_disease.joblib")

@app.route("/")
def home():
    return "Heart Disease Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract relevant fields from request data
        nmv = float(data.get("ca", 0))
        tcp = float(data.get("cp", 0))
        eia = float(data.get("exang", 0))
        thal = float(data.get("thal", 1))
        op = float(data.get("oldpeak", 0))
        mhra = float(data.get("thalach", 0))
        age = float(data.get("age", 0))

        # Create input array and make prediction
        input_array = np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1)
        prediction = heart_model.predict(input_array)

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/predict_heart', methods=['POST'])
def api_predict_heart():
    try:
        data = request.get_json()
        nmv = float(data.get("ca", 0))
        tcp = float(data.get("cp", 0))
        eia = float(data.get("exang", 0))
        thal = float(data.get("thal", 1))
        op = float(data.get("oldpeak", 0))
        mhra = float(data.get("thalach", 0))
        age = float(data.get("age", 0))

        input_array = np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1)
        prediction = heart_model.predict(input_array)

        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
