# CDC Diabetes Risk Predictor

A machine learning web application that predicts diabetes risk using an ensemble of XGBoost, CatBoost, and Neural Network models trained on the CDC Diabetes Health Indicators dataset.

---

## Features

- Interactive web UI with 21 health indicators
- Four prediction modes: XGBoost, CatBoost, Neural Network, and Ensemble
- Weighted ensemble: XGBoost (35%) + CatBoost (40%) + Neural Network (25%)
- Real-time risk visualization with probability bars and risk verdicts
- REST API for programmatic access
- Comprehensive test suite (17 test cases)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask, Gunicorn |
| Models | XGBoost, CatBoost, scikit-learn MLPClassifier |
| Preprocessing | scikit-learn, imbalanced-learn |
| Frontend | Vanilla HTML/CSS/JavaScript |
| Deployment | Docker, Render |
| Testing | pytest |

---

## Getting Started

### Prerequisites

- Python 3.12+
- pip

### Local Development

```bash
# Clone the repository
git clone <repo-url>
cd CDC_Diabetes_Prediction

# Install dependencies
pip install -r requirements.txt

# Start the development server
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### Docker

```bash
# Build the image (runs tests automatically during build)
docker build -t cdc-diabetes-predictor .

# Run the container
docker run -p 5000:5000 cdc-diabetes-predictor
```

### Run Tests

```bash
pytest tests/
```

---

## API Reference

### `GET /health`
Health check.

```json
{ "status": "ok" }
```

### `POST /predict/<model>`
Predict with a single model. `model` is one of: `xgboost`, `catboost`, `neural_network`, `ensemble`.

**Request body** — JSON object with 21 health features (see [Input Features](#input-features)).

**Response:**
```json
{
  "model": "xgboost",
  "probability": 0.72,
  "prediction": 1
}
```

### `POST /predict`
Predict with all four models simultaneously.

**Response:**
```json
{
  "xgboost":        { "probability": 0.72, "prediction": 1 },
  "catboost":       { "probability": 0.68, "prediction": 1 },
  "neural_network": { "probability": 0.65, "prediction": 1 },
  "ensemble":       { "probability": 0.69, "prediction": 1 }
}
```

---

## Input Features

### Binary (0 or 1)

| Feature | Description |
|---|---|
| HighBP | High blood pressure |
| HighChol | High cholesterol |
| CholCheck | Cholesterol check in last 5 years |
| Smoker | Smoked ≥100 cigarettes in lifetime |
| Stroke | Ever had a stroke |
| HeartDiseaseorAttack | Coronary heart disease or heart attack |
| PhysActivity | Physical activity in past 30 days |
| Fruits | Consumes fruit ≥1 time per day |
| Veggies | Consumes vegetables ≥1 time per day |
| HvyAlcoholConsump | Heavy alcohol consumption |
| AnyHealthcare | Has any healthcare coverage |
| NoDocbcCost | Couldn't see a doctor due to cost |
| DiffWalk | Difficulty walking or climbing stairs |
| Sex | Sex (0 = Female, 1 = Male) |

### Ordinal / Continuous

| Feature | Range | Description |
|---|---|---|
| BMI | continuous | Body mass index |
| MentHlth | 0–30 | Days of poor mental health (past 30 days) |
| PhysHlth | 0–30 | Days of poor physical health (past 30 days) |
| Age | 1–13 | Age group (1 = 18–24, 13 = 80+) |
| Education | 1–6 | Education level |
| Income | 1–8 | Income bracket |
| GenHlth | 1–5 | Self-rated general health (1 = Excellent, 5 = Poor) |

### Engineered Features (auto-generated)

| Feature | Description |
|---|---|
| BMI_Category | BMI binned into 4 categories (underweight/normal/overweight/obese) |
| BMI_HighBP | BMI × HighBP interaction |
| Age_GenHlth | Age × GenHlth interaction |
| PhysAct_BMI | PhysActivity × BMI interaction |

---

## Model Training

Training code is in [Mlmodel.ipynb](Mlmodel.ipynb). The notebook:

1. Fetches the [CDC Diabetes Health Indicators dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) from the UCI ML Repository
2. Performs feature engineering
3. Trains XGBoost, CatBoost, and MLP models with Optuna hyperparameter tuning
4. Handles class imbalance with imbalanced-learn
5. Serializes trained models and preprocessor to `models/`

---

## Deployment

This app is configured for deployment on [Render](https://render.com) via [render.yaml](render.yaml).

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
- **Python version:** 3.12.0

The Dockerfile validates the test suite at build time — a failing test will prevent the image from being built.

---

## Project Structure

```
CDC_Diabetes_Prediction/
├── app.py                # Flask application & prediction logic
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker build config
├── render.yaml           # Render deployment config
├── runtime.txt           # Python version pin
├── Mlmodel.ipynb         # Model training notebook
├── models/
│   ├── xgboost_model.pkl
│   ├── catboost_model.pkl
│   ├── mlp_model.pkl
│   └── preprocessor.pkl
├── templates/
│   └── index.html        # Web UI
└── tests/
    └── test_app.py       # Test suite
```

---

## License

This project uses the [CDC Diabetes Health Indicators dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) sourced from the UCI Machine Learning Repository.
