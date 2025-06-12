# 🎯 Sistem Rekomendasi CV

Sistem ini membantu HR/rekruter untuk merekomendasikan kandidat terbaik berdasarkan kecocokan skill, pengalaman, dan kebutuhan posisi kerja secara otomatis menggunakan algoritma machine learning hybrid.

## 🚀 Fitur Utama

- **Rekomendasi Hybrid:** Kombinasi content-based, collaborative filtering, dan skills matching.
- **Dashboard Interaktif:** UI modern berbasis Streamlit, mudah digunakan.
- **Manajemen Kandidat:** Tambah, edit, dan kelola database kandidat.
- **Analisis & Laporan:** Visualisasi performa, analisis gap skill, dan export laporan.
- **Explainable AI:** Penjelasan transparan untuk setiap skor rekomendasi.

## 🗂️ Struktur Proyek

```
.
├── streamlit_app.py              # Aplikasi web utama (Streamlit)
├── cv_system.py                  # Engine ML & logika rekomendasi
├── run_app.py                    # Script startup otomatis
├── requirements.txt              # Daftar dependensi Python
├── data_kandidat.csv             # Contoh data kandidat
├── cv_recommendation_model.pkl   # Model terlatih (opsional)
└── README.md                     # Dokumentasi proyek
```

## ⚙️ Instalasi & Menjalankan

### 1. **Persiapan Lingkungan**
```bash
python -m venv venv
# Aktifkan venv:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. **Instalasi Dependensi**
```bash
pip install -r requirements.txt
```

### 3. **Menjalankan Aplikasi**
```bash
# Cara 1: Otomatis
python run_app.py

# Cara 2: Manual
streamlit run streamlit_app.py
```

## 📋 Cara Penggunaan

1. **Buka aplikasi** di browser (biasanya http://localhost:8501).
2. **Upload model** atau gunakan model default.
3. **Pilih posisi pekerjaan** dan dapatkan rekomendasi kandidat terbaik.
4. **Tambah kandidat baru** melalui form di dashboard.
5. **Lihat analitik** dan laporan performa rekrutmen.

## 🧩 Dependensi Utama

- Python >= 3.8
- Streamlit
- Pandas
- Numpy
- Scikit-learn
- Plotly

## 📄 Lisensi

MIT License

**Catatan:**  
- Pastikan file `data_kandidat.csv` tersedia untuk menjalankan aplikasi.
- Model terlatih (`cv_recommendation_model.pkl`) opsional, bisa melatih ulang dari dashboard. 