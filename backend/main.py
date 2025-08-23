from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
# --- TAMBAHAN PENTING ---
from pycaret.regression import load_model, predict_model

# --- Konfigurasi Aplikasi ---
app = FastAPI(
    title="API Prediksi Harga Properti",
    description="API untuk memprediksi harga properti menggunakan model ML.",
    version="1.0"
)

# --- Pemuatan Model ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_price_prediction_pipeline.pkl") 

model = None

@app.on_event("startup")
def startup_event():
    """Memuat pipeline model saat aplikasi pertama kali dijalankan."""
    global model
    try:
        model = load_model(MODEL_PATH.replace('.pkl','')) # PyCaret butuh nama tanpa .pkl
        print(f"Model berhasil dimuat dari {MODEL_PATH}")
    except FileNotFoundError:
        print(f"PERINGATAN: File model tidak ditemukan di {MODEL_PATH}.")
    except Exception as e:
        print(f"Terjadi error saat memuat model: {e}")

# --- Struktur Input & Mapping Bulan (Tetap sama) ---
class HousingFeatures(BaseModel):
    neighborhood: str
    area: float
    construction_year: float
    rooms_count: float
    floor: float
    unit_per_floor: float | None = None
    has_elevator: bool
    has_parking: bool
    has_warehouse: bool
    year: int
    month: str

MONTH_MAP = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11, "desember": 12
}

# --- Endpoints API ---
@app.get("/")
def read_root():
    return {"status": "API Prediksi Harga Properti Aktif!"}

@app.post("/predict")
def predict_price(features: HousingFeatures):
    """Menerima data fitur properti dan mengembalikan prediksi harga dalam USD."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model tidak tersedia atau gagal dimuat.")

    try:
        data = features.dict()
        month_str = data['month'].lower()
        if month_str not in MONTH_MAP:
            raise HTTPException(status_code=400, detail=f"Nama bulan tidak valid: {data['month']}.")
        data['month'] = MONTH_MAP[month_str]

        input_df = pd.DataFrame([data])

        prediction_df = predict_model(model, data=input_df)
        predicted_price = prediction_df['prediction_label'].iloc[0]

        return {"predicted_price_usd": float(predicted_price)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi error saat prediksi: {str(e)}")
