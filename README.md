# Car Price Predictor

**AutoValue AI** is a machine-learning web application for estimating used-car
selling prices. Users enter vehicle specifications to receive an estimate in
Indian Rupees (INR), review estimates saved in their browser, compare saved
vehicles, and explore model evaluation results.

> **Academic project:** B.Tech Minor Project, Department of Computer Science
> and Engineering, Sanskriti University, Mathura (Academic Session 2023–2027).

## Contents

- [Features](#features)
- [Technology](#technology)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Run locally](#run-locally)
- [Deploy to Render](#deploy-to-render)
- [API](#api)
- [Model and evaluation](#model-and-evaluation)
- [Tests](#tests)
- [Limitations](#limitations)
- [Academic project](#academic-project)

## Features

- Predicts used-car prices from vehicle name, year, kilometres driven, fuel,
  seller type, transmission, and ownership history.
- Displays estimated prices in INR and explains that estimates are not
  guaranteed transaction prices.
- Saves successful predictions in browser local storage.
- Search, reuse, and delete saved prediction records.
- Select and compare up to four saved vehicles.
- Displays model metrics, feature importance, price-band errors, and
  actual-versus-predicted evaluation visualizations.
- Responsive React interface for desktop and mobile screens.

## Technology

| Area | Technologies |
| --- | --- |
| Frontend | React, JavaScript, Vite, CSS |
| Backend | Python, Flask, Flask-CORS |
| Machine learning | pandas, NumPy, scikit-learn, joblib |
| Data and notebooks | Jupyter, matplotlib, seaborn |

## How it works

1. The React application collects and validates the vehicle details.
2. The frontend sends a JSON request to the Flask API through Vite's `/api`
   development proxy.
3. Flask validates the request and passes the input to the saved
   scikit-learn pipeline.
4. The API returns the estimated price and model information.
5. The frontend displays the estimate and, on this browser, saves it to
   prediction history.

The production pipeline is the **Log-target Extra Trees** model stored in
`models/car_price_pipeline.joblib`.

## Project structure

```text
Car-Price-Predictor/
├── backend/
│   ├── app.py
│   ├── evaluate_models.py
│   ├── evaluate_saved_model.py
│   ├── requirements.txt
│   └── test_app.py
├── data/
│   ├── raw/
│   │   └── CAR DETAILS FROM CAR DEKHO.csv
│   └── processed/
│       └── car_price_cleaned.csv
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── models/
│   ├── car_price_pipeline.joblib
│   ├── evaluation.json
│   └── model_metadata.json
├── notebooks/
├── requirements.txt
└── README.md
```

The data and model artifacts listed above are present in the development
workspace. If you distribute the project, check dataset and model artifact
redistribution permissions and include any required source attribution.

## Requirements

- Python 3.10 or later
- Node.js and npm
- Windows PowerShell, Command Prompt, macOS, or Linux terminal

## Run locally

Run the backend and frontend in separate terminals from the project root.

### 1. Set up and start the backend

Create and activate a virtual environment, then install the backend
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

Start the Flask API:

```powershell
python backend\app.py
```

The API listens at `http://127.0.0.1:5000`.

For macOS or Linux, activate the environment with
`source .venv/bin/activate`. The backend requirements can be installed with
`python -m pip install -r backend/requirements.txt`.

### 2. Set up and start the frontend

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`. Keep both
development servers running while using the app. Vite forwards frontend
requests from `/api/*` to the Flask server and removes the `/api` prefix.

## Deploy to Render

Deploy the backend as a **Web Service** and the frontend as a **Static Site**.
Both services use the same GitHub repository and the `main` branch.

### 1. Deploy the backend

Create a Render Web Service with these settings:

| Setting | Value |
| --- | --- |
| Root Directory | Leave blank (repository root) |
| Build Command | `pip install -r backend/requirements.txt` |
| Start Command | `gunicorn backend.app:app --bind 0.0.0.0:$PORT` |
| Health Check Path | `/health` |

After deployment, open `https://<your-backend-service>.onrender.com/health`.
Confirm the response reports `"model_loaded": true`. The prediction model at
`models/car_price_pipeline.joblib` must be available in the repository.

### 2. Deploy the frontend

Create a Render Static Site from the same repository:

| Setting | Value |
| --- | --- |
| Root Directory | `frontend` |
| Build Command | `npm ci && npm run build` |
| Publish Directory | `dist` |

Add this environment variable to the static site:

| Variable | Value |
| --- | --- |
| `VITE_API_URL` | `https://<your-backend-service>.onrender.com` |

Use your actual backend service URL, with no trailing slash. Vite embeds this
variable during the frontend build, so save it and redeploy the static site
after adding or changing it. The frontend uses this URL for `/predict` and
`/evaluation`. If `VITE_API_URL` is unset, local development continues to use
Vite's `/api` proxy.

Once both services are deployed, open the static-site URL, confirm evaluation
data loads, and submit a prediction to verify frontend-to-backend requests.

### Optional: install notebook and analysis dependencies

The root `requirements.txt` includes the data-analysis and notebook
dependencies. Install it into the activated Python environment if you plan to
run the exploratory notebooks:

```powershell
python -m pip install -r requirements.txt
```

### Regenerate model evaluation

The evaluation script reads `data/processed/car_price_cleaned.csv` and writes
the evaluation and model artifacts under `models/`. Run it from the project
root:

```powershell
python backend\evaluate_models.py
```

This script trains and evaluates candidate models; running it can replace the
saved model and metadata artifacts. The API can serve the checked-in evaluation
with `GET /evaluation` without rerunning training.

## API

The Flask API accepts and returns JSON.

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | API status and endpoint information |
| `GET` | `/health` | Backend and model availability |
| `POST` | `/predict` | Estimate a vehicle's selling price |
| `GET` | `/evaluation` | Read the saved model evaluation |

### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Log-target Extra Trees"
}
```

### Price prediction

```http
POST /predict
Content-Type: application/json
```

Request:

```json
{
  "name": "Maruti Swift Dzire VDI",
  "year": 2017,
  "km_driven": 60000,
  "fuel": "Diesel",
  "seller_type": "Individual",
  "transmission": "Manual",
  "owner": "First Owner"
}
```

Example response (the price shown is illustrative):

```json
{
  "predicted_price": 621878.22,
  "currency": "INR",
  "model": "Log-target Extra Trees",
  "note": "Estimate only; actual selling prices depend on vehicle condition, location and market conditions."
}
```

All seven request fields are required. The API validates the JSON object,
required values, integer year and mileage, and non-negative kilometres. It
returns an appropriate HTTP error when the request is invalid or the model is
unavailable.

> The Vite proxy adds the `/api` prefix in frontend development. Requests from
> the React app use `/api/predict` and `/api/evaluation`; direct API requests
> to Flask use `/predict` and `/evaluation`.

## Model and evaluation

The application uses a scikit-learn pipeline trained on used-car data. The
saved evaluation compares Extra Trees, Random Forest, and Gradient Boosting
using a shared 80/20 train-test split (random state 42).

| Candidate model | Test MAE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: |
| Extra Trees | ₹1,65,636 | ₹4,21,322 | 0.4489 |
| Random Forest | ₹1,74,157 | ₹4,04,521 | 0.4920 |
| Gradient Boosting | ₹1,64,914 | ₹4,20,107 | 0.4521 |

The current production pipeline is **Log-target Extra Trees**. Although
Gradient Boosting had the lowest MAE in the recorded comparison, it did not
meet the project's model replacement threshold, so Extra Trees remains the
production model. Metrics are from the project's shared holdout comparison;
because that split participates in model selection, it is not an untouched
final test set.

- **MAE** is the average absolute difference between actual and predicted
  prices.
- **RMSE** gives larger errors more weight.
- **R²** summarizes the proportion of price variation explained by the model.

The evaluation dashboard also highlights price-band differences. In
particular, estimates for expensive or uncommon vehicle variants can be less
reliable. Model estimates should be interpreted with these limitations in
mind.

## Tests

With the backend dependencies installed and the virtual environment active,
run the Flask API unit tests from the project root:

```powershell
python -m unittest discover -s backend -p "test_*.py"
```

Check the frontend build and lint rules:

```powershell
cd frontend
npm run build
npm run lint
```

## Limitations

- The estimate is based on historical data and is not a guaranteed sale price,
  appraisal, or financial recommendation.
- The model does not directly include vehicle condition, accident or service
  history, location, or current market demand.
- Rare variants and high-priced vehicles may be underrepresented in the
  training data and have larger errors.
- Prediction history is stored in the browser's local storage; it is not
  synced to an account or another device.
- Model performance depends on the coverage, quality, and freshness of the
  dataset.

## Academic project

AutoValue AI is a B.Tech Minor Project for the Department of Computer Science
and Engineering at Sanskriti University, Mathura, for Academic Session
2026–2027. The project is guided by **Mr. Deep Sen** and developed by a team of
four students.

### Acknowledgements

We thank Sanskriti University, the Department of Computer Science and
Engineering, and our project guide for their support. We also acknowledge the
maintainers of the open-source tools and libraries used in this project.

### Disclaimer

This application was developed for academic and demonstration purposes. Its
outputs are machine-learning estimates derived from historical data and should
not replace a professional vehicle inspection or independent market research.
