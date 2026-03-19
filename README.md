# EV Detection System — Real Govt Data

A machine learning project that predicts whether a registered vehicle in Telangana is electric or not, built using real government data from data.telangana.gov.in.

---

## What this project does

I built a system that takes basic vehicle details — manufacturer, body type, engine CC, and RTA office — and predicts whether the vehicle is electric. The model is trained on 2.8 million real vehicle registration records from the Telangana Transport Department and served through a REST API with a simple web UI.

---

## Why I built this

I wanted to work with real messy data instead of a clean Kaggle tutorial dataset. The Telangana government publishes monthly vehicle registration data openly, so I used that. While exploring the data I discovered something interesting — EV adoption in Hyderabad grew from 0.017% of all registrations in 2018 to 8.66% in 2026. That's a 500x growth in 8 years. That trend became the core story of this project.

---

## The hardest part

The biggest challenge was that the older CSV files and newer monthly files had completely different column names. For example the fuel column was called `Fuel` in one schema and `fuel` in another, and maker was `Maker_Name` in one and `makerName` in another. I had to write a column normalization layer before merging all 8 files together.

The second problem was data quality — registration dates ranged from 1753 to 2075 due to data entry errors. I had to define a valid date range filter before any time series analysis.

---

## What I learned about my model

When I first trained the model I got 99% F1 score and thought it was great. Then I got suspicious — it was too perfect. I ran a controlled experiment where I removed the manufacturer name feature completely and retrained. Precision dropped from 98% to 91%.

This told me the model was partially relying on manufacturer identity — Ola Electric always maps to electric, Honda always maps to petrol. But the bigger finding was that even without manufacturer, the model stayed strong because CC=0 is the dominant signal. Electric vehicles have no combustion engine so they register with zero engine capacity. The model learned a real physical relationship, not just a brand shortcut.

---

## Tech stack

- Python, pandas, scikit-learn
- FastAPI + Uvicorn
- HTML/CSS/JS frontend
- Data source: data.telangana.gov.in (Transport Department)

---

## Project structure

```
ml_project/
├── data/                  # Raw CSV files from Telangana govt
├── model/                 # Saved model and encoders
├── templates/             # Frontend UI
├── read.py                # Data loading, cleaning, training
├── api.py                 # FastAPI backend
└── README.md
```

---

## How to run

```bash
# Install dependencies
pip install fastapi uvicorn scikit-learn pandas joblib jinja2

# Train the model
python read.py

# Start the API
uvicorn api:app --reload

# Open UI
http://127.0.0.1:8000/ui
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Predict if vehicle is electric |
| GET | `/makers` | List all known manufacturers |
| GET | `/body-types` | List all body types |
| GET | `/offices` | List all RTA offices |
| GET | `/ui` | Web interface |

---

## Sample prediction

```json
POST /predict
{
  "maker_name": "OLA ELECTRIC TECHNOLOGIES PVT LTD",
  "body_type": "Solo",
  "cc": 0,
  "office": "RTA UPPAL"
}

Response:
{
  "is_electric": true,
  "confidence": 1.0,
  "message": "Electric vehicle"
}
```

---

## Key finding

EV adoption in Telangana by year:

| Year | EV share |
|------|----------|
| 2018 | 0.017% |
| 2021 | 0.544% |
| 2022 | 2.289% |
| 2023 | 5.235% |
| 2025 | 7.687% |
| 2026 | 8.660% |

---

Built by a fresher who actually opened the data instead of just planning to.
