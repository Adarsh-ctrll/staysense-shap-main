# StaySense — NYC Airbnb Room Type Predictor

StaySense predicts whether an NYC Airbnb listing is an **Entire home/apt**, **Private room**, or **Shared room**, using pricing, availability, review activity, host behavior, and location — and explains *why* using SHAP.

**Live demo:** https://staysense-shap.onrender.com
> Hosted on Render's free tier — the API may take up to ~60 seconds to wake up on first request.

---

## What it does

Given a set of listing details, StaySense:

- Predicts the room type with a trained Random Forest classifier
- Returns class probabilities for all three room types
- Explains the prediction with SHAP, showing which features pushed the prediction toward (or away from) the predicted class
- Generates a downloadable PDF report of the prediction and explanation
- Keeps a local history of recent predictions in the browser

## How it works

```
listing details → FastAPI /predict → Random Forest → room type + probabilities
                                 ↘ /explain → SHAP TreeExplainer → per-feature contributions
                                 ↘ /report  → PDF summary (ReportLab)
```

The frontend is a static HTML/CSS/JS app served directly by FastAPI, so there's no separate frontend build or deployment — the same service serves the UI and the API.

## Tech stack

| Layer | Tools |
|---|---|
| Model | Scikit-learn (Random Forest), imbalanced-learn (SMOTE) |
| Explainability | SHAP (`TreeExplainer`) |
| Backend | FastAPI, Pydantic, Uvicorn |
| Reports | ReportLab (PDF generation) |
| Frontend | HTML, CSS, JavaScript, Chart.js |

## Model

The classifier is trained on the [NYC Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) dataset. Training, feature engineering, and evaluation are documented in [`nyc_airbnb_room_type_predictor.ipynb`](./nyc_airbnb_room_type_predictor.ipynb).


- **Features:** `latitude`, `longitude`, `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`, `neighbourhood_group`, `neighbourhood`
- **Class imbalance:** handled with SMOTE during training, since Shared Room listings are a small minority of the dataset
- **Model selection:** class-weighted Random Forest, tuned with `RandomizedSearchCV` on Macro F1
- **Performance:** ~85% accuracy, ~73.5% Macro F1 on held-out data (per-class performance is weaker on the minority Shared Room class — see the notebook for the full breakdown)
- The trained pipeline (preprocessing + classifier) is serialized to `model_pipeline.pkl` and loaded directly by the API — it is not retrained at request time.

## API reference

Base URL: `/` (same origin as the frontend)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend (`index.html`) |
| `GET` | `/health` | Health check |
| `GET` | `/meta` | Model metadata — classes, raw feature names, transformed feature count |
| `POST` | `/predict` | Predicted room type + class probabilities |
| `POST` | `/explain` | SHAP explanation for the prediction |
| `POST` | `/report` | Downloadable PDF report (prediction + explanation) |

`/predict`, `/explain`, and `/report` all accept the same JSON body:

```json
{
  "latitude": 40.7484,
  "longitude": -73.9857,
  "price": 120,
  "minimum_nights": 2,
  "number_of_reviews": 84,
  "reviews_per_month": 2.3,
  "calculated_host_listings_count": 1,
  "availability_365": 210,
  "neighbourhood_group": "Manhattan",
  "neighbourhood": "Midtown"
}
```

Example `/predict` response:

```json
{
  "Predicted_room_type": "Entire home/apt",
  "Probability": [0.71, 0.24, 0.05]
}
```

Interactive API docs (Swagger UI) are available at `/docs` once the server is running.

## Running locally

Requires Python 3.12.

```bash
git clone https://github.com/Adarsh-ctrll/staysense-shap.git
cd staysense-shap

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

uvicorn main_shap:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

## Project structure

```
.
├── main_shap.py        # FastAPI app: routes, prediction, SHAP, report generation
├── shap_utils.py        # SHAP explainer + feature grouping logic
├── report_generator.py  # PDF report generation (ReportLab)
├── model_pipeline.pkl    # Trained Scikit-learn pipeline (preprocessing + classifier)
├── index.html / style.css / script.js   # Frontend
├── report.html          # Standalone visual report page
├── nyc_airbnb_room_type_predictor.ipynb # Data analysis, training, evaluation
└── requirements.txt
```

## Notes

- `model_pipeline.pkl` is a pre-trained artifact — the API loads it as-is and does not retrain on startup.
- This is a portfolio/demo project, not a production compliance or fraud-detection tool. Predictions and SHAP explanations are illustrative and shouldn't be used as the sole basis for real-world listing or regulatory decisions.
