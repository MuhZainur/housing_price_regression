# Tahap 1: Pilih Fondasi
# Kita mulai dengan base image Python 3.10 versi "slim" yang ringan dan efisien.
FROM python:3.10-slim

# Tahap 2: Siapkan Ruang Kerja
# Membuat dan pindah ke direktori /app di dalam kontainer.
WORKDIR /app

# Tahap 3: Salin "Daftar Belanjaan"
# Salin file requirements.txt terlebih dahulu. Docker sangat pintar,
# jika file ini tidak berubah, ia tidak akan mengulang instalasi di build selanjutnya.
COPY requirements.txt .

# Tahap 4: Instal Semua Kebutuhan
# Jalankan 'pip install' untuk semua library dari requirements.txt.
# Opsi --no-cache-dir membuat ukuran image kita lebih kecil.
RUN pip install --no-cache-dir -r requirements.txt

# Tahap 5: Salin Seluruh Isi "Dapur"
# Salin semua file lain dari folder proyek kita (main.py, dan nanti file model .pkl)
# ke dalam direktori /app di dalam kontainer.
COPY . .

# Tahap 6: Beritahu Dunia Pintu Mana yang Dibuka
# Memberitahu Docker bahwa aplikasi kita di dalam akan berjalan di port 8080.
# Google Cloud Run secara default akan menggunakan port ini.
EXPOSE 8080

# Tahap 7: Perintah untuk Menjalankan Aplikasi
# Ini adalah perintah yang akan dijalankan saat kontainer dimulai.
# Ia akan menyalakan server Uvicorn untuk menjalankan aplikasi FastAPI kita.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
