import argparse
import json
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)
MODEL = None
SCALER = None

def load_model(model_path):
    global MODEL, SCALER
    MODEL = tf.keras.models.load_model(model_path)
    try:
        SCALER = joblib.load(f"{model_path}/scaler.joblib")
    except Exception:
        SCALER = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    instances = payload.get('instances')
    if instances is None:
        return jsonify({'error': 'provide "instances" list'}), 400
    arr = np.array(instances, dtype=float)
    if SCALER is not None:
        arr = SCALER.transform(arr)
    preds = MODEL.predict(arr)
    preds_list = preds.tolist()
    return jsonify({'predictions': preds_list})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/my_model', help='model directory')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    load_model(args.model)
    app.run(host=args.host, port=args.port)

if __name__ == '__main__':
    main()
