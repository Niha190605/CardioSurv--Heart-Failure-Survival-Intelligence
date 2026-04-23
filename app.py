from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib, numpy as np, pandas as pd
import os

app = FastAPI(title="CardioSurv API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ──────────────────────────────────────────────
BASE = os.path.dirname(__file__)
rsf    = joblib.load(os.path.join(BASE, "rsf_model.pkl"))
scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))

FEATURE_COLS = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
    'ejection_fraction', 'high_blood_pressure', 'platelets',
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking',
    'age_group', 'ef_risk', 'creatinine_risk', 'sodium_risk',
    'comorbidity_score', 'age_ef_interaction', 'log_cpk', 'log_platelets'
]

# ── Input schema ─────────────────────────────────────────────────────────────
class Patient(BaseModel):
    age: float
    anaemia: int
    creatinine_phosphokinase: float
    diabetes: int
    ejection_fraction: float
    high_blood_pressure: int
    platelets: float
    serum_creatinine: float
    serum_sodium: float
    sex: int
    smoking: int

# ── Feature engineering (exact copy from notebook Cell 14 + Gradio cell) ─────
def engineer_features(p: dict) -> dict:
    p = p.copy()
    age = p['age']
    p['age_group']          = 0 if age < 50 else 1 if age < 60 else 2 if age < 70 else 3
    p['ef_risk']            = int(p['ejection_fraction'] < 40)
    p['creatinine_risk']    = int(p['serum_creatinine'] > 1.5)
    p['sodium_risk']        = int(p['serum_sodium'] < 135)
    p['comorbidity_score']  = (p['anaemia'] + p['diabetes'] +
                                p['high_blood_pressure'] + p['smoking'])
    p['age_ef_interaction'] = p['age'] * p['ejection_fraction']
    p['log_cpk']            = np.log1p(p['creatinine_phosphokinase'])
    p['log_platelets']      = np.log1p(p['platelets'])
    return p

# ── Predict endpoint ──────────────────────────────────────────────────────────
@app.post("/predict")
def predict(patient: Patient):
    d = engineer_features(patient.dict())
    X    = pd.DataFrame([d])[FEATURE_COLS]
    X_sc = scaler.transform(X)

    risk_score = float(rsf.predict(X_sc)[0])
    surv_fn    = rsf.predict_survival_function(X_sc)[0]

    # Safe timepoints within the RSF model's training domain (0–271 days)
    max_t = int(surv_fn.domain[1])
    timepoints = [t for t in [30, 60, 90, 120, 150, 180, 210, 240, 260] if t <= max_t]
    surv_probs = {str(t): round(float(surv_fn(t)), 4) for t in timepoints}
    t_6m = min(180, max_t)
    mortality_6m = round(1 - float(surv_fn(t_6m)), 4)

    # Normalise risk score to 0-1 for the gauge (RSF scores are relative)
    # Clamp between typical observed range 0-20
    mortality_prob = round(min(max(mortality_6m, 0.0), 1.0), 4)

    return {
        "risk_score":    round(risk_score, 4),
        "mortality_prob": mortality_prob,
        "survival_probs": surv_probs,
        "flags": {
            "ef_risk":         bool(d['ef_risk']),
            "creatinine_risk": bool(d['creatinine_risk']),
            "sodium_risk":     bool(d['sodium_risk']),
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# ── Serve HTML frontend ───────────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse(os.path.join(BASE, "index.html"))
