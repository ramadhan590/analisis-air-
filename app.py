import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide",
)

sns.set_style("whitegrid")

@st.cache_data(show_spinner=True)
def load_data():
    # 1. Gunakan path absolut yang dinamis berdasarkan lokasi script ini berjalan
    current_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = Path(current_dir) / "data" 
    
    # 2. Cari file dengan ekstensi .csv mengabaikan huruf besar/kecil
    csv_files = [f for f in BASE_DIR.rglob("*") if f.suffix.lower() == '.csv']
    csv_files = sorted(csv_files)
    
    if not csv_files:
        st.error(f"Gagal! Tidak ada file CSV ditemukan di: {BASE_DIR.absolute()}")
        st.info("💡 Pastikan folder 'data' tidak ter-ignore oleh .gitignore dan sudah ter-push ke GitHub.")
        st.stop()

    df_list = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)

            # 3. Proteksi KeyError: Buat kolom station jika hilang
            if "station" not in df.columns or df["station"].isna().all():
                # Ambil nama stasiun dari nama file
                parts = file_path.stem.split("_")
                station_name = parts[2] if len(parts) > 2 else file_path.stem
                df["station"] = station_name
            
            df_list.append(df)
        except Exception as e:
            st.warning(f"File {file_path.name} bermasalah: {e}")

    # Cegah error jika semua file gagal dibaca
    if not df_list:
        st.error("Semua file CSV gagal diproses.")
        st.stop()

    # Gabungkan semua
    df = pd.concat(df_list, ignore_index=True)

    # 4. Proses Datetime
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]],
        errors="coerce"
    )

    # 5. Bersihkan data
    df = df.dropna(subset=["datetime"])
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)

    # 6. PERBAIKAN: Isi missing value (ffill & bfill) yang dijamin aman
    # Hanya menargetkan kolom numerik agar kolom string ('station') tidak terhapus Pandas
    cols_to_fill = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    for col in cols_to_fill:
        if col in df.columns:
            # Gunakan transform agar bentuk DataFrame tidak berubah dan kolom station aman
            df[col] = df.groupby("station")[col].transform(lambda x: x.ffill().bfill())

    return df

def filter_data(df):
    stations = sorted(df["station"].dropna().unique().tolist())
    selected_stations = st.sidebar.multiselect(
        "Pilih stasiun",
        options=stations,
        default=stations,
    )

    min_date = df["datetime"].min().date()
    max_date = df["datetime"].max().date()

    selected_dates = st.sidebar.date_input(
        "Rentang tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = min_date, max_date

    filtered = df[
        (df["station"].isin(selected_stations)) &
        (df["datetime"].dt.date >= start_date) &
        (df["datetime"].dt.date <= end_date)
    ].copy()

    return filtered, selected_stations, start_date, end_date

# --- FUNGSI PLOTTING & ANALISIS (DISESUAIKAN DENGAN NOTEBOOK) ---

def plot_seasonal_trend(df):
    # Data Preparation: Rata-rata bulanan
    seasonal_df = df.groupby(df['datetime'].dt.month)['PM2.5'].mean().reset_index()
    seasonal_df.columns = ['month', 'PM2.5']

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=seasonal_df, x='month', y='PM2.5', marker='o', linewidth=3, color='darkred', ax=ax)
    ax.fill_between(seasonal_df['month'], seasonal_df['PM2.5'], color="red", alpha=0.1)

    ax.set_title("Tren Musiman Konsentrasi PM2.5 (Berdasarkan Filter)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Bulan", fontsize=12)
    ax.set_ylabel("Rata-rata PM2.5 (µg/m³)", fontsize=12)
    ax.set_xticks(range(1, 13))
    
    st.pyplot(fig)
    return seasonal_df

def plot_station_comparison(df):
    # Threshold batas aman
    THRESHOLD = 75
    
    # Menghitung Rata-rata dan % Pelanggaran
    station_avg = df.groupby('station')['PM2.5'].mean().sort_values(ascending=False).reset_index()
    breach_count = df.groupby('station')['PM2.5'].apply(lambda x: (x > THRESHOLD).sum() / len(x) * 100).reset_index()
    breach_count.columns = ['station', 'breach_percentage']

    # Penggabungan data stasiun
    station_stats = pd.merge(station_avg, breach_count, on='station').sort_values(by='PM2.5', ascending=False)

    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Plot Bar untuk Rata-rata
    sns.barplot(data=station_stats, x='station', y='PM2.5', palette='viridis', hue='station', legend=False, ax=ax1)
    ax1.set_ylabel('Rata-rata PM2.5 (µg/m³)', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Plot Line untuk % Pelanggaran
    ax2 = ax1.twinx()
    sns.lineplot(data=station_stats, x='station', y='breach_percentage', color='red', marker='D', linewidth=2, ax=ax2)
    ax2.set_ylabel('Persentase Melampaui Ambang Batas (%)', fontsize=12, color='red', fontweight='bold')

    plt.title("Perbandingan Kualitas Udara & Konsistensi Polusi Antar Stasiun", fontsize=16, fontweight='bold')
    st.pyplot(fig)
    
    return station_stats

def cluster_air_quality(pm_value):
    if pm_value <= 35:
        return 'Baik'
    elif pm_value <= 75:
        return 'Sedang'
    elif pm_value <= 150:
        return 'Tidak Sehat'
    else:
        return 'Sangat Tidak Sehat'

def plot_advanced_analysis(df):
    df_clean = df.copy()
    df_clean['quality_category'] = df_clean['PM2.5'].apply(cluster_air_quality)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(
        data=df_clean, 
        x='quality_category', 
        order=['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat'], 
        palette='RdYlGn_r',
        hue='quality_category',
        legend=False,
        ax=ax
    )
    ax.set_title('Distribusi Frekuensi Kategori Kualitas Udara', fontsize=14, fontweight='bold')
    ax.set_xlabel('Kategori', fontsize=12)
    ax.set_ylabel('Jumlah Observasi', fontsize=12)
    
    st.pyplot(fig)
    
    quality_distribution = df_clean['quality_category'].value_counts(normalize=True) * 100
    return quality_distribution

# --- MAIN APP ---

def main():
    st.title("🌫️ Dashboard Analisis Data: Air Quality")
    st.markdown(
        """
        Dashboard ini menyajikan hasil analisis data kualitas udara yang diadaptasi dari proyek Jupyter Notebook. 
        Analisis ini berfokus pada pemahaman pola musiman dan tingkat polusi antar stasiun pengamatan.
        """
    )

    st.sidebar.header("Filter Data")

    df = load_data()

    with st.expander("Cek Detail Dataset (Data Raw)"):
        st.write("Jumlah seluruh observasi:", f"{len(df):,}")
        st.write("Daftar Stasiun:", sorted(df["station"].dropna().unique().tolist()))
        st.dataframe(df.head())

    filtered_df, selected_stations, start_date, end_date = filter_data(df)

    if filtered_df.empty:
        st.warning("Tidak ada data yang sesuai filter.")
        return

    # Menampilkan Filter Metrik Singkat
    st.subheader("Ringkasan Data Saat Ini")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Observasi", f"{len(filtered_df):,}")
    c2.metric("Stasiun Terpilih", filtered_df["station"].nunique())
    c3.metric("Rata-rata PM2.5 (Filter)", f"{filtered_df['PM2.5'].mean():.2f} µg/m³")
    c4.metric("PM2.5 Tertinggi (Filter)", f"{filtered_df['PM2.5'].max():.2f} µg/m³")
    st.divider()

    # --- PERTANYAAN 1 ---
    st.subheader("Pertanyaan 1: Pola Musiman Konsentrasi PM2.5")
    st.markdown("*Apakah terdapat pola musiman yang konsisten pada konsentrasi PM2.5 di seluruh stasiun pengamatan, dan pada bulan-bulan apa polusi mencapai titik paling kritis?*")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        seasonal_df = plot_seasonal_trend(filtered_df)
    with col2:
        st.markdown("**Tabel Rata-rata PM2.5 Bulanan**")
        st.dataframe(seasonal_df.style.format({"PM2.5": "{:.2f}"}), use_container_width=True)

    st.info("""
    **Insight Pola Musiman:**
    Terdapat fluktuasi polusi yang sangat dipengaruhi oleh perubahan bulan/musim dengan siklus 'U-shaped'.
    * **Titik Tertinggi:** Polusi mencapai titik paling kritis pada bulan Desember dan Januari. Ini berhubungan dengan musim dingin di mana penggunaan pemanas ruangan meningkat dan kondisi atmosfer menjebak polutan.
    * **Titik Terendah:** Kualitas udara paling bersih ditemukan pada bulan Agustus yang bertepatan dengan musim panas dengan curah hujan tinggi.
    """)
    st.divider()


    # --- PERTANYAAN 2 ---
    st.subheader("Pertanyaan 2: Perbandingan Kualitas Udara Antar Stasiun")
    st.markdown("*Bagaimana perbandingan kualitas udara antar stasiun pengamatan, dan stasiun mana yang paling sering melampaui ambang batas (75 µg/m³) secara konsisten?*")
    
    station_stats = plot_station_comparison(filtered_df)
    
    st.markdown("**Tabel Statistik per Stasiun (Rata-rata & % Pelanggaran)**")
    st.dataframe(station_stats.style.format({"PM2.5": "{:.2f}", "breach_percentage": "{:.2f}%"}), use_container_width=True)

    st.info("""
    **Insight Perbandingan Stasiun:**
    * **Stasiun Paling Berpolusi:** Stasiun **Dongsi** dan **Wanshouxigong** memiliki rata-rata PM2.5 tertinggi serta memiliki persentase pelanggaran ambang batas paling sering (>40%).
    * **Stasiun Paling Bersih:** Stasiun **Dingling** menunjukkan performa kualitas udara yang jauh lebih stabil dan bersih dibanding stasiun lain, dengan persentase pelanggaran yang paling minim.
    """)
    st.divider()


    # --- ANALISIS LANJUTAN ---
    st.subheader("Analisis Lanjutan: Kategorisasi & Binning Risiko")
    
    col_adv1, col_adv2 = st.columns([1, 1])
    
    with col_adv1:
        st.markdown("**Distribusi Kategori Kualitas Udara (Manual Clustering)**")
        quality_distribution = plot_advanced_analysis(filtered_df)
        dist_df = pd.DataFrame(quality_distribution).reset_index()
        dist_df.columns = ['Kategori', 'Persentase (%)']
        st.dataframe(dist_df.style.format({"Persentase (%)": "{:.2f}%"}), use_container_width=True)
        
    with col_adv2:
        st.markdown("**Pengelompokan Risiko Stasiun (Binning Analysis)**")
        # Mengelompokkan stasiun berdasarkan rentang resiko
        station_risk = station_stats[['station', 'PM2.5']].copy()
        # Hindari error out of bounds menggunakan rentang max yang aman
        max_pm25 = station_risk['PM2.5'].max() + 10 
        station_risk['risk_level'] = pd.cut(
            station_risk['PM2.5'],
            bins=[0, 75, 85, max_pm25],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        st.dataframe(station_risk.sort_values(by='PM2.5', ascending=False).style.format({"PM2.5": "{:.2f}"}), use_container_width=True)

    st.divider()


    # --- KESIMPULAN ---
    st.subheader("📌 Conclusion")
    st.markdown("""
    Berdasarkan hasil analisis data kualitas udara dari stasiun pengamatan:

    1. **Pola Musiman yang Konsisten:** Kualitas udara sangat dipengaruhi oleh siklus tahunan. Polusi PM2.5 mencapai titik paling kritis pada **bulan Desember dan Januari** (Musim Dingin). Sebaliknya, kualitas udara paling bersih ditemukan pada **bulan Agustus** (Musim Panas).
    2. **Ketimpangan Kualitas Udara Antar Wilayah:** Tidak semua wilayah memiliki beban polusi yang sama. Stasiun **Dongsi** teridentifikasi sebagai wilayah dengan **Risiko Tinggi (High Risk)**, sedangkan stasiun **Dingling** secara konsisten menjadi wilayah yang paling bersih (**Low Risk**).
    3. **Status Kesehatan Udara Secara Keseluruhan:** Meskipun kategori 'Baik' cukup besar, terdapat porsi yang sangat signifikan (sekitar **~39%**) di mana kualitas udara berada pada level **'Tidak Sehat' hingga 'Sangat Tidak Sehat'**. Ini menunjukkan penduduk sering kali terpapar polusi yang melampaui batas aman.
    4. **Rekomendasi Kebijakan:** Intervensi pengurangan polusi harus diprioritaskan pada stasiun-stasiun di kategori 'High Risk' (seperti Dongsi, Wanshouxigong) dan dilakukan pengetatan kontrol emisi terutama menjelang akhir tahun untuk memitigasi lonjakan polusi musiman.
    """)

if __name__ == "__main__":
    main()
