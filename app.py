import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load models and preprocessor
with open('models/xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open('models/catboost_model.pkl', 'rb') as f:
    catboost_model = pickle.load(f)
with open('models/mlp_model.pkl', 'rb') as f:
    mlp_model = pickle.load(f)
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

BINARY_FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex'
]

RANGE_FEATURES = [
    'BMI', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income', 'GenHlth'
]

def engineer_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100],
                                labels=[0, 1, 2, 3]).astype(float)
    df['BMI_HighBP']  = df['BMI'] * df['HighBP']
    df['Age_GenHlth'] = df['Age'] * df['GenHlth']
    df['PhysAct_BMI'] = df['PhysActivity'] * df['BMI']
    return df

def predict(df: pd.DataFrame, model_name: str) -> dict:
    X = preprocessor.transform(df)
    if model_name in ('xgboost', 'ensemble'):
        prob_xgb = float(xgb_model.predict_proba(X)[:, 1][0])
    if model_name in ('catboost', 'ensemble'):
        prob_cat = float(catboost_model.predict_proba(X)[:, 1][0])
    if model_name in ('neural_network', 'ensemble'):
        prob_mlp = float(mlp_model.predict_proba(X)[:, 1][0])

    if model_name == 'xgboost':
        prob = prob_xgb
    elif model_name == 'catboost':
        prob = prob_cat
    elif model_name == 'neural_network':
        prob = prob_mlp
    else:
        prob = prob_xgb * 0.35 + prob_cat * 0.40 + prob_mlp * 0.25

    return {'probability': round(prob, 4), 'prediction': int(prob >= 0.5)}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict/<model_name>', methods=['POST'])
def predict_route(model_name):
    valid = ('xgboost', 'catboost', 'neural_network', 'ensemble')
    if model_name not in valid:
        return jsonify({'error': f"model_name must be one of {valid}"}), 400

    body = request.get_json(force=True)
    required = BINARY_FEATURES + RANGE_FEATURES
    missing = [f for f in required if f not in body]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        df = engineer_features(body)
        result = predict(df, model_name)
        result['model'] = model_name
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict_all():
    body = request.get_json(force=True)
    required = BINARY_FEATURES + RANGE_FEATURES
    missing = [f for f in required if f not in body]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        df = engineer_features(body)
        results = {m: predict(df, m) for m in ('xgboost', 'catboost', 'neural_network', 'ensemble')}
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
