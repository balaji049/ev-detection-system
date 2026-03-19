from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI(title="EV Classifier API")
templates = Jinja2Templates(directory="templates")

@app.get("/ui")
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

model     = joblib.load("model/ev_classifier.pkl")
le_maker  = joblib.load("model/le_maker.pkl")
le_body   = joblib.load("model/le_body.pkl")
le_office = joblib.load("model/le_office.pkl")

class VehicleInput(BaseModel):
    maker_name: str
    body_type: str
    cc: float
    office: str

class PredictionOutput(BaseModel):
    is_electric: bool
    confidence: float
    message: str

@app.get("/")
def root():
    return {"status": "running", "model": "EV Classifier v1"}

@app.post("/predict", response_model=PredictionOutput)
def predict(vehicle: VehicleInput):
    try:
        maker_enc  = le_maker.transform([vehicle.maker_name])[0]
        body_enc = le_body.transform([vehicle.body_type])[0]
        office_enc = le_office.transform([vehicle.office])[0]
    except ValueError as e:
        return PredictionOutput(
            is_electric=False,
            confidence=0.0,
            message=f"Unknown value: {str(e)}"
        )

    X = np.array([[maker_enc, body_enc, vehicle.cc, office_enc]])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    prob = proba[1] if pred == 1 else proba[0]

    return PredictionOutput(
        is_electric=bool(pred),
        confidence=round(float(prob), 4),
        message="Electric vehicle" if pred else "Non-electric vehicle"
    )

@app.get("/makers")
def get_makers():
    return {"makers": list(le_maker.classes_)}

@app.get("/body-types")
def get_body_types():
    return {"body_types": list(le_body.classes_)}

@app.get("/offices")
def get_offices():
    return {"offices": list(le_office.classes_)}