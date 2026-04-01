# EV Detection System
[![Dataset Size](https://img.shields.io/badge/Data-2.8M%20Records-blue)]()
[![EV Growth](https://img.shields.io/badge/EV%20Growth-500x-green)]()
> Machine learning classifier trained on 2.8 million real vehicle registration records from the Telangana government. Discovered EV adoption in Hyderabad grew from **0.017% to 8.66%** between 2018 and 2026.

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?style=flat-square)](https://scikit-learn.org)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=flat-square)](https://ev-detection-system-dneb.onrender.com/ui)

**[Live Demo](https://ev-detection-system-dneb.onrender.com/ui)** | **[Dataset](https://data.telangana.gov.in)** | **[LinkedIn](https://linkedin.com/in/balaji-bhairwad)**

---

## What this project does

This project predicts whether a registered vehicle in Telangana is electric or petrol/diesel, using only basic registration details: manufacturer name, body type, engine CC, and RTA office.

The model is trained on **2.8 million real vehicle registration records** from the Telangana Transport Department open data portal and served through a FastAPI REST endpoint with a plain HTML/JS web interface.

This is not a Kaggle tutorial dataset. It is real government data with all the problems that come with it.

---

## The data problem

The Telangana Transport Department publishes monthly registration CSV files. Each batch had different column names:

| Field | Old schema | New schema |
|-------|-----------|------------|
| Fuel type | `Fuel` | `fuel` |
| Manufacturer | `Maker_Name` | `makerName` |
| Registration date | `Reg_Date` | `regDate` |

I wrote a column normalization layer before merging all 8 files. On top of that, registration dates ranged from **1753 to 2075** due to data entry errors at RTA offices. I defined a valid date range filter before any time series analysis could begin.

---

## The 99% F1 problem

After training, the model scored 99% F1. Instead of publishing that number, I got suspicious.

I ran a controlled ablation: removed the manufacturer name feature entirely and retrained.

**Result:** Precision dropped from 98% to 91%.

This confirmed the model was partially relying on manufacturer identity. Ola Electric always maps to electric. Honda always maps to petrol. But the bigger finding was that even without manufacturer names, the model stayed strong at 91% because **CC=0 is the dominant signal**. Electric vehicles have no combustion engine so they register with zero engine capacity. The model learned a real physical relationship, not a brand shortcut.

A 99% F1 score on a clean split is not impressive. Understanding *why* the model works is.

---

## Key finding: EV adoption in Telangana

| Year | EV share of all registrations |
|------|-------------------------------|
| 2018 | 0.017% |
| 2021 | 0.544% |
| 2022 | 2.289% |
| 2023 | 5.235% |
| 2025 | 7.687% |
| 2026 | 8.660% |

**That is a 500x increase in 8 years.** This trend became the core story of the project.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Data processing | Python, pandas |
| Machine learning | scikit-learn |
| API | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Data source | Telangana Transport Dept open data portal |
| Deployment | Render |

---

## Project structure

```
ml_project/
├── data/         # Raw CSV files from Telangana government
├── model/        # Saved model and label encoders (joblib)
├── templates/    # Frontend HTML/JS UI
├── read.py       # Data loading, normalization, training pipeline
├── api.py        # FastAPI backend with prediction endpoint
└── README.md
```

---

## How to run locally

```bash
# Clone the repo
git clone https://github.com/balaji049/ev-detection-system
cd ev-detection-system

# Install dependencies
pip install fastapi uvicorn scikit-learn pandas joblib jinja2

# Train the model (reads from data/ folder)
python read.py

# Start the API server
uvicorn api:app --reload

# Open the web interface
# http://127.0.0.1:8000/ui
```

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Predict if a vehicle is electric |
| GET | `/makers` | List all known manufacturers |
| GET | `/body-types` | List all body types |
| GET | `/offices` | List all RTA offices |
| GET | `/ui` | Web interface |

### Sample request

```json
POST /predict
{
  "maker_name": "OLA ELECTRIC TECHNOLOGIES PVT LTD",
  "body_type": "Solo",
  "cc": 0,
  "office": "RTA UPPAL"
}
```

### Sample response

```json
{
  "is_electric": true,
  "confidence": 1.0,
  "message": "Electric vehicle"
}
```

---

## What I would add next

- Visualisation dashboard showing EV adoption by district and RTA office
- Monthly trend charts updated with new government data releases
- Feature importance plot from the trained model
- Comparison of pre and post ablation confusion matrices

---

## Data source

All data is sourced from the **Telangana Transport Department** via the state open data portal at [data.telangana.gov.in](https://data.telangana.gov.in). The dataset is publicly available under the Government of Telangana open data license.

---

## Author

**Balaji Bhairwad**
B.Tech, Computer Science and Engineering (AI and ML)
Malla Reddy Engineering College, Hyderabad

[LinkedIn](https://linkedin.com/in/balaji-bhairwad) | [GitHub](https://github.com/balaji049) | [Live Demo](https://ev-detection-system-dneb.onrender.com/ui)

---

*Built by a fresher who actually opened the data instead of just planning to.*
