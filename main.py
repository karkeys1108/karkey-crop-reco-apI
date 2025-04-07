from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

# Load models and encoders
model = joblib.load("model/crop_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
rice_df = pd.read_csv("data/RICE_TNAU_STXT.csv")

class CropRequest(BaseModel):
    temperature: float
    humidity: float
    ph: float
    district: str
    month: str
    soil_texture: str

@app.post("/predict/")
def predict_crop(req: CropRequest):
    # Predict top 5 crops
    input_df = pd.DataFrame([[req.temperature, req.humidity, req.ph]], columns=['temperature', 'humidity', 'ph'])
    proba = model.predict_proba(input_df)[0]
    crops = label_encoder.inverse_transform(np.arange(len(proba)))
    crop_confidence = list(zip(crops, proba * 100))
    top5 = sorted(crop_confidence, key=lambda x: x[1], reverse=True)[:5]

    result = {"Top_5_Crops": [{c[0]: f"{c[1]:.2f}%"} for c in top5]}

    # Rice variety suggestion
    if any(c[0].lower() == "rice" for c in top5):
        month = req.month.strip().lower().capitalize()
        district = req.district.strip().lower()
        texture = req.soil_texture.strip().lower()

        rice_df['TNDST'] = rice_df['TNDST'].str.strip().str.lower()
        rice_df['STXT'] = rice_df['STXT'].str.strip().str.lower()
        rice_df['STMT'] = rice_df['STMT'].str.strip().str.lower().str.capitalize()
        rice_df['EDMT'] = rice_df['EDMT'].str.strip().str.lower().str.capitalize()

        filtered = rice_df[
            (rice_df['TNDST'] == district) &
            (rice_df['STXT'] == texture) &
            (rice_df['STMT'] <= month) &
            (rice_df['EDMT'] >= month)
        ]

        varieties = filtered['VRTS'].unique().tolist()
        result["Rice_Varieties"] = varieties if varieties else ["No varieties found"]
    return result
