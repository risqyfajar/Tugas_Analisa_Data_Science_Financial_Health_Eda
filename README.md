# Analisis Kesehatan Keuangan Korporasi & Prediksi Kebangkrutan
## Proyek Sains Data — Tugas Ujian Tengah Semester (UTS)

### 📌 Ringkasan Proyek
Proyek ini melakukan analisis eksplorasi data (*Exploratory Data Analysis* / EDA) komprehensif untuk mendeteksi secara dini tingkat kesehatan finansial perusahaan dan risiko kebangkrutan korporasi (*Corporate Financial Distress & Bankruptcy Prediction*). Analisis ini menggunakan kerangka kerja standar industri **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) yang dipetakan secara runtut dari **Bab 1 hingga Bab 9** sesuai panduan resmi pelaporan UTS.

---

### 📂 Dataset Riil (`DATA_SHEET/data.csv`)
Dataset yang digunakan adalah data riil publikasi resmi:
* **Sumber:** *Taiwan Economic Journal (TEJ)* (periode 1999–2009) via Kaggle.
* **Jumlah Observasi:** 6.819 perusahaan.
* **Jumlah Fitur:** 96 kolom (95 rasio keuangan + 1 variabel target).
* **Variabel Target:** `Bankrupt?`
  * Kelas `0` (Perusahaan Sehat / Solven): 6.599 perusahaan (96.77%)
  * Kelas `1` (Perusahaan Bangkrut / Distress): 220 perusahaan (3.23%)
  * *Tantangan:* **Heavy Imbalanced Class** (~3.23% kelas minoritas).

---

### 🗂 Struktur Berkas di Repositori
```text
d:\PASCA\SMT 1\DATA SIENCE\TUGAS\
├── DATA_SHEET/
│   └── data.csv                  <- Dataset riil Taiwan Bankruptcy (6.819 baris, 96 kolom)
├── financial_health_eda.ipynb    <- Jupyter Notebook lengkap (22 cell terstruktur Bab 1 s/d 9)
├── financial_health_eda.py       <- Skrip Python terintegrasi untuk terminal/IDE
└── README.md                     <- Dokumentasi proyek dan panduan teknis
```

---

### 🚀 Cara Menjalankan Notebook (Tahap 1)

#### Opsi A: Google Colab (Paling Praktis)
1. Buka [Google Colab](https://colab.research.google.com).
2. Klik menu **File > Upload Notebook**, lalu pilih berkas `financial_health_eda.ipynb`.
3. Buat folder bernama `DATA_SHEET` di panel Files Colab, lalu unggah berkas `data.csv` ke dalam folder tersebut (atau unggah langsung `data.csv` ke root panel Files).
4. Klik **Runtime > Run all** untuk mengeksekusi seluruh 22 sel secara otomatis.

#### Opsi B: Menggunakan Editor Lokal (VS Code / JupyterLab)
1. Buka folder kerja `d:\PASCA\SMT 1\DATA SIENCE\TUGAS\` di VS Code atau JupyterLab.
2. Buka berkas `financial_health_eda.ipynb`.
3. Jalankan sel kode satu per satu untuk mengeksplorasi visualisasi data.

---

### 📊 Pemetaan Bab Laporan (Bab 1 s/d Bab 9)
* **Bab 1: Pemahaman Bisnis (*Business Understanding*)**
  * Latar belakang mitigasi risiko kredit korporasi, problem statement, tipe klasifikasi biner, dan penetapan metrik teknis (**Recall** $\ge 0.85$ dan **ROC-AUC** $\ge 0.88$).
* **Bab 2: Pemahaman Data (*Data Understanding*)**
  * Taksonomi 95 rasio keuangan: profitabilitas, likuiditas, solvabilitas, arus kas, dan variabel target `Bankrupt?`.
* **Bab 3: Memuat & Memeriksa Data (*Importing & Inspecting*)**
  * Pemuatan dataset dari folder `DATA_SHEET`, *whitespace stripping* nama kolom, pembersihan kolom konstan (`Net Income Flag` varian nol), dan ringkasan statistik.
* **Bab 4: Penanganan Data yang Hilang (*Handling Missing Data*)**
  * Audit integritas data (Dataset 100% lengkap tanpa missing values).
* **Bab 5: Eksplorasi Karakteristik & Distribusi Data**
  * Analisis *Skewness* dan *Kurtosis* fitur kunci, histogram, dan kurva KDE perbandingan sebaran.
* **Bab 6: Transformasi Data & Rekayasa Fitur (*Feature Engineering*)**
  * Pembuatan rasio komposit: *Debt-to-ROA Ratio* dan *Stability Score*.
* **Bab 7: Visualisasi Hubungan Data (Bivariat & Multivariat)**
  * Identifikasi ketimpangan kelas 3.23%, komparasi boxplot Sehat vs Bangkrut, dan Matriks Korelasi Spearman.
* **Bab 8: Deteksi & Penanganan Pencilan (*Outliers*)**
  * Deteksi metode Rentang Interkuartil (IQR) dan aplikasi **Winsorization (Capping)** tanpa menghilangkan data krisis.
* **Bab 9: Kesimpulan, Temuan Utama & Rekomendasi Pemodelan**
  * Rangkuman temuan empiris, justifikasi teknik SMOTE / Class Weight, dan rekomendasi model ensemble berbasis pohon (*XGBoost / Random Forest*).
