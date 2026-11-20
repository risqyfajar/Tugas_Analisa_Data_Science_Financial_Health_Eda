"""
================================================================================
PROYEK SAINS DATA (UTS PASCASARJANA)
STUDI KASUS: ANALISIS KESEHATAN KEUANGAN KORPORASI & PREDIKSI KEBANGKRUTAN
(Taiwan Economic Journal - Company Bankruptcy Prediction)
Metodologi: CRISP-DM | Bab 1 s/d Bab 9 Sesuai Panduan Resmi
Dataset: DATA_SHEET/data.csv (6.819 Baris, 96 Kolom)
================================================================================
"""

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# BAB 3: MEMUAT & MEMERIKSA DATA
# ------------------------------------------------------------------------------
print("="*75)
print("BAB 3: MEMUAT & MEMERIKSA DATA DARI FOLDER DATA_SHEET")
print("="*75)

path_opsi = [
    os.path.join("DATA_SHEET", "data.csv"),
    os.path.join("..", "DATA_SHEET", "data.csv"),
    "data.csv"
]

data_path = None
for p in path_opsi:
    if os.path.exists(p):
        data_path = p
        break

if not data_path:
    raise FileNotFoundError("File data.csv di folder DATA_SHEET tidak ditemukan!")

df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()
target_col = 'Bankrupt?'

print(f"[OK] Memuat data dari: {data_path}")
print(f"Dimensi Data: {df.shape[0]:,} baris, {df.shape[1]} kolom")

# Eliminasi kolom konstan jika ada
kolom_konstan = [c for c in df.columns if df[c].nunique() <= 1]
if kolom_konstan:
    print(f"Eliminasi kolom konstan: {kolom_konstan}")
    df.drop(columns=kolom_konstan, inplace=True)

# ------------------------------------------------------------------------------
# BAB 4: PENANGANAN DATA HILANG (MISSING VALUES)
# ------------------------------------------------------------------------------
print("\n" + "="*75)
print("BAB 4: AUDIT INTEGRITAS & MISSING VALUES")
print("="*75)
missing_total = df.isnull().sum().sum()
print(f"Total Nilai Hilang (Missing Values): {missing_total}")
if missing_total == 0:
    print("Dataset lengkap (100% complete dataset tanpa missing values).")

# ------------------------------------------------------------------------------
# BAB 5: EKSPLORASI KARAKTERISTIK & SKEWNESS
# ------------------------------------------------------------------------------
print("\n" + "="*75)
print("BAB 5: EKSPLORASI KARAKTERISTIK & SKEWNESS FITUR KUNCI")
print("="*75)

fitur_fokus = [
    'ROA(C) before interest and depreciation before interest',
    'Debt ratio %',
    'Net worth/Assets',
    'Persistent EPS in the Last Four Seasons',
    'Working Capital to Total Assets',
    'Net Income to Total Assets',
    'Cash/Total Assets'
]

skew_df = pd.DataFrame({
    'Mean': df[fitur_fokus].mean(),
    'Median': df[fitur_fokus].median(),
    'Skewness': df[fitur_fokus].skew(),
    'Kurtosis': df[fitur_fokus].kurtosis()
})
print(skew_df)

# ------------------------------------------------------------------------------
# BAB 7: VISUALISASI HUBUNGAN DATA & MATRIKS KORELASI
# ------------------------------------------------------------------------------
print("\n" + "="*75)
print("BAB 7: DISTRIBUSI TARGET & KORELASI SPEARMAN")
print("="*75)

target_counts = df[target_col].value_counts()
print("Proporsi Target (Bankrupt?):")
print(f" - Sehat (0)   : {target_counts[0]:,} ({target_counts[0]/len(df)*100:.2f}%)")
print(f" - Bangkrut (1): {target_counts[1]:,} ({target_counts[1]/len(df)*100:.2f}%)")
print("--> Kesimpulan: Imbalanced Class Ekstrem (~3.23% Bangkrut)")

corr_spearman = df.corr(method='spearman')[target_col].drop(target_col)
print("\nTop 5 Fitur Berkorelasi Positif (Memicu Kebangkrutan):")
print(corr_spearman.sort_values(ascending=False).head(5))

print("\nTop 5 Fitur Berkorelasi Negatif (Proteksi Finansial):")
print(corr_spearman.sort_values(ascending=True).head(5))

# ------------------------------------------------------------------------------
# BAB 8: PENANGANAN PENCILAN (OUTLIERS)
# ------------------------------------------------------------------------------
print("\n" + "="*75)
print("BAB 8: DETEKSI & PENANGANAN PENCILAN (METODE IQR)")
print("="*75)

for col in fitur_fokus:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr
    n_out = len(df[(df[col] < lb) | (df[col] > ub)])
    print(f"Fitur: {col[:35]:<35} | Outliers: {n_out:4d} ({(n_out/len(df))*100:4.1f}%) | Bounds: [{lb:.2f}, {ub:.2f}]")
    # Winsorization / Capping
    df[col] = np.clip(df[col], lb, ub)

print("\n[OK] Winsorization (IQR Capping) selesai diaplikasikan.")

# ------------------------------------------------------------------------------
# BAB 9: KESIMPULAN & REKOMENDASI PEMODELAN
# ------------------------------------------------------------------------------
print("\n" + "="*75)
print("BAB 9: KESIMPULAN & REKOMENDASI PEMODELAN MACHINE LEARNING")
print("="*75)
print("1. ROA, Net Income to Total Assets, dan Persistent EPS adalah prediktor")
print("   paling kuat dengan korelasi negatif signifikan terhadap kebangkrutan.")
print("2. Beban utang (Debt ratio %) melonjak signifikan pada perusahaan bangkrut.")
print("3. Penanganan ketimpangan kelas (3.23% bangkrut) wajib menggunakan SMOTE / class_weight.")
print("4. Algoritma rekomendasi: Tree Ensemble (XGBoost, Random Forest) dengan metrik RECALL.")
print("="*75)
