import pandas as pd
import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# --- Konfigurasi Aplikasi ---
app = FastAPI(
    title="API Prediksi Harga Properti",
    description="API untuk memprediksi harga properti berdasarkan fitur-fiturnya.",
    version="1.0"
)

# --- Pemuatan Model (Akan diaktifkan nanti) ---
# Dapatkan path direktori saat ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_price_prediction_pipeline.pkl") # Diasumsikan nama modelnya ini

model = None

# @app.on_event("startup")
# def load_model():
#     """Memuat pipeline model saat aplikasi pertama kali dijalankan."""
#     global model
#     try:
#         model = joblib.load(MODEL_PATH)
#         print(f"Model berhasil dimuat dari {MODEL_PATH}")
#     except FileNotFoundError:
#         print(f"PERINGATAN: File model tidak ditemukan di {MODEL_PATH}. Endpoint /predict tidak akan bekerja.")
#     except Exception as e:
#         print(f"Terjadi error saat memuat model: {e}")

# --- Struktur Input Data ---
# Disesuaikan persis dengan kolom dan tipe data Anda
class HousingFeatures(BaseModel):
    neighborhood: str
    area: float
    construction_year: float
    rooms_count: float
    floor: float
    unit_per_floor: Optional[float] = None # Opsional karena ada nilai NaN
    has_elevator: bool
    has_parking: bool
    has_warehouse: bool
    year: int
    month: str # User akan memasukkan nama bulan, misal: "Januari"

# --- Mapping Bulan (Sesuai Permintaan Anda) ---
MONTH_MAP = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11, "desember": 12
}

# --- Endpoints API ---
@app.get("/")
def read_root():
    return {"status": "API Prediksi Harga Properti Aktif!"}

# Endpoint ini akan kita gunakan setelah modelnya jadi
@app.post("/predict")
def predict_price(features: HousingFeatures):
    """
    Menerima data fitur properti dan mengembalikan prediksi harga dalam USD.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model tidak tersedia atau gagal dimuat.")

    try:
        # 1. Ubah input Pydantic menjadi dictionary
        data = features.dict()

        # 2. Konversi nama bulan menjadi angka
        month_str = data['month'].lower()
        if month_str not in MONTH_MAP:
            raise HTTPException(status_code=400, detail=f"Nama bulan tidak valid: {data['month']}. Gunakan nama bulan dalam Bahasa Indonesia.")
        data['month'] = MONTH_MAP[month_str]

        # 3. Buat DataFrame dengan satu baris
        input_df = pd.DataFrame([data])

        # 4. Lakukan prediksi
        # PyCaret's predict_model akan mengembalikan DataFrame dengan kolom 'prediction_label'
        prediction_df = predict_model(model, data=input_df)
        predicted_price = prediction_df['prediction_label'].iloc[0]

        return {"predicted_price_usd": float(predicted_price)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi error saat prediksi: {str(e)}")
