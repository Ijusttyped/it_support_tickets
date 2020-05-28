import pickle

import pandas as pd
from flask import Flask, jsonify, request

from modeling.models import model_predict

app = Flask(__name__)
# CORS(app)

with open("../models/BayesianRidge.pkl", "rb") as pkl:
    model = pickle.load(pkl)


@app.route("/api/v1.0/duration", methods=["POST"])
def predict():
    req = request.json
    data = pd.read_json(req)
    predictions = model_predict(to_predict=data, model=model)
    predictjson = predictions.to_json()
    return jsonify(predictjson)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
