from fastapi import FastAPI
from typing import Any, Dict
import pandas as pd
from catboost import CatBoostClassifier
import json, joblib
from pathlib import Path

app = FastAPI()


# --- load once at startup ---

root = Path(__file__).resolve().parent
model = CatBoostClassifier()
model.load_model(root / "catboost_final.cbm")

with open(root / "catboost_artifacts.json", "r") as f:
    artifacts = json.load(f)

feature_order = artifacts["feature_order"]
cat_cols = artifacts["cat_cols"]
feature_maps = joblib.load(root / "feature_maps.joblib")

# --- endpoints ---
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Dict[str, Any]):
    REQUIRED_FIELDS = ["UID","M1","M2","M4","M5","M6","M7","M9","P_emaildomain"]
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return {"error": f"Missing required fields: {missing}"}

    if "UID" not in payload:
        return {"error": "UID is required"}

    # 1) add engineered features from feature_maps
    for feat_name, series in feature_maps.items():
        nlevels = getattr(series.index, "nlevels", 1)

        if nlevels == 1:
            val = series.get(payload["UID"], 0.0)
        else:
            second_key_col = feat_name.split("_UID_")[0]  # e.g., "M1"
            second_val = payload.get(second_key_col)
            val = series.get((payload["UID"], second_val), 0.0)

        payload[feat_name] = 0.0 if val is None else float(val)

    # 2) dataframe + enforce columns/order
    df = pd.DataFrame([payload])
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_order]

    # 3) cast categoricals to string
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)

    df = df.fillna(0.0)

    # 4) predict
    prob = float(model.predict_proba(df)[0, 1])
    pred = int(prob >= 0.5)
    return {
    "model": "catboost_final",
    "threshold": 0.5,
    "prob": prob,
    "pred": pred
}



# http://127.0.0.1:8000/

# http://127.0.0.1:8000/docs