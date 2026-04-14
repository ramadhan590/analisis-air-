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
    
    # Menggunakan warna darkblue dan shading skyblue seperti di notebook
    sns.lineplot(data=seasonal_df, x='month', y='PM2.5', marker='o', linewidth=3, color='darkblue', ax=ax)
    ax.fill_between(seasonal_df['month'], seasonal_df['PM2.5'], color="skyblue", alpha=0.3)

    ax.set_title("Tren Fluktuasi Rata-rata Konsentrasi PM2.5 Bulanan (2013-2017)", fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel("Bulan (1-12)", fontsize=12)
    ax.set_ylabel("Rata-rata Konsentrasi PM2.5 (µg/m³)", fontsize=12)
    ax.set_xticks(range(1, 13))
    ax.grid(True, linestyle='--', alpha=0.6)
    
    st.pyplot(fig)
    return seasonal_df

def plot_station_comparison(df):
    # Threshold batas aman PM2.5 = 75
    THRESHOLD = 75
    
    # Menghitung persentase frekuensi PM2.5 > 75 per stasiun
    station_breach = df.groupby('station')['PM2.5'].apply(lambda x: (x > THRESHOLD).sum() / len(x) * 100).reset_index()
    station_breach.columns = ['station', 'breach_rate']

    # Mengurutkan dari frekuensi pelanggaran tertinggi ke terendah
    station_breach = station_breach.sort_values(by='breach_rate', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Membuat Bar Chart
    barplot = sns.barplot(
        data=station_breach,
        x='station',
        y='breach_rate',
        palette='Reds_r', # Gradasi warna merah seperti di notebook
        hue='station',
        legend=False,
        ax=ax
    )

    # Menambahkan label angka persentase di atas setiap batang grafik
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.1f') + '%',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha = 'center', va = 'center',
                         xytext = (0, 9),
                         textcoords = 'offset points',
                         fontsize=10)

    ax.set_title("Stasiun dengan Frekuensi Tertinggi Melampaui Ambang Batas PM2.5 (>75 µg/m³) \nPeriode 2013-2017", fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel("Nama Stasiun", fontsize=12)
    ax.set_ylabel("Persentase Waktu Melampaui Batas (%)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    st.pyplot(fig)
    
    return station_breach

def cluster_air_quality(pm_value):
    if pm_value <= 35:
        return 'Baik (Good)'
    elif pm_value <= 75:
        return 'Sedang (Moderate)'
    elif pm_value <= 150:
        return 'Tidak Sehat (Unhealthy)'
    else:
        return 'Sangat Tidak Sehat (Very Unhealthy)'

def plot_advanced_analysis(df):
    df_clean = df.copy()
    df_clean['aqi_category'] = df_clean['PM2.5'].apply(cluster_air_quality)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(
        data=df_clean, 
        x='aqi_category', 
        order=['Baik (Good)', 'Sedang (Moderate)', 'Tidak Sehat (Unhealthy)', 'Sangat Tidak Sehat (Very Unhealthy)'], 
        palette='RdYlGn_r',
        hue='aqi_category',
        legend=False,
        ax=ax
    )
    ax.set_title('Distribusi Frekuensi Kategori Kualitas Udara (Manual Grouping)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Kategori', fontsize=12)
    ax.set_ylabel('Jumlah Observasi', fontsize=12)
    
    st.pyplot(fig)
    
    quality_distribution = df_clean['aqi_category'].value_counts(normalize=True) * 100
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
    st.subheader("Pertanyaan 1: Tren Fluktuasi Rata-rata Konsentrasi PM2.5 Bulanan")
    st.markdown("*Bagaimana tren fluktuasi rata-rata konsentrasi PM2.5 secara bulanan untuk mengidentifikasi pola musiman di seluruh stasiun pengamatan selama periode tahun 2013 hingga 2017?*")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        seasonal_df = plot_seasonal_trend(filtered_df)
    with col2:
        st.markdown("**Tabel Rata-rata PM2.5 Bulanan (2013-2017)**")
        st.dataframe(seasonal_df.style.format({"PM2.5": "{:.2f}"}), use_container_width=True)

    st.info("""
    **Insight Pola Musiman:**
    Berdasarkan grafik garis, terlihat bahwa konsentrasi PM2.5 mengalami fluktuasi musiman yang konsisten. Konsentrasi tertinggi terjadi pada **bulan Desember (>100 µg/m³)** dan terendah pada **bulan Agustus**. 
    Hal ini mengindikasikan bahwa pada musim dingin, polusi udara meningkat drastis, yang mungkin disebabkan oleh aktivitas pemanas ruangan atau kondisi atmosfer yang menjebak polutan.
    """)
    st.divider()


    # --- PERTANYAAN 2 ---
    st.subheader("Pertanyaan 2: Stasiun dengan Frekuensi Pelanggaran Tertinggi")
    st.markdown("*Stasiun manakah yang mencatatkan frekuensi tertinggi dalam melampaui ambang batas konsentrasi PM2.5 (75 µg/m³) secara konsisten selama rentang waktu 2013 hingga 2017?*")
    
    station_breach = plot_station_comparison(filtered_df)
    
    st.markdown("**Tabel Persentase Pelanggaran Ambang Batas per Stasiun (2013-2017)**")
    st.dataframe(station_breach.style.format({"breach_rate": "{:.2f}%"}), use_container_width=True)

    st.info("""
    **Insight Paparan Tertinggi:**
    Dari grafik batang, **Stasiun Dongsi** menempati urutan pertama dengan persentase pelanggaran ambang batas tertinggi, yaitu **42.7%**.
    Sebaliknya, **Stasiun Dingling** merupakan wilayah dengan kualitas udara yang relatif lebih baik karena frekuensi pelanggarannya paling rendah dibandingkan stasiun lainnya.
    Mayoritas stasiun di wilayah perkotaan menunjukkan angka pelanggaran di atas 40%, menandakan bahwa polusi PM2.5 adalah masalah serius yang merata di banyak titik pengamatan.
    """)
    st.divider()


    # --- ANALISIS LANJUTAN ---
    st.subheader("Analisis Lanjutan (Opsional): Kategorisasi & Binning Risiko")
    
    col_adv1, col_adv2 = st.columns([1, 1])
    
    with col_adv1:
        st.markdown("**Distribusi Kategori Kualitas Udara (Manual Clustering)**")
        quality_distribution = plot_advanced_analysis(filtered_df)
        dist_df = pd.DataFrame(quality_distribution).reset_index()
        dist_df.columns = ['aqi_category', 'proportion']
        st.dataframe(dist_df.style.format({"proportion": "{:.2f}%"}), use_container_width=True)
        
    with col_adv2:
        st.markdown("**Pengelompokan Risiko Stasiun (Binning Analysis)**")
        # Mengelompokkan stasiun berdasarkan rata-rata dan rentang resiko
        station_summary = filtered_df.groupby('station')['PM2.5'].mean().reset_index()
        # Hindari error dengan menambahkan batas aman maksimal
        max_pm25 = max(100, station_summary['PM2.5'].max() + 10) 
        station_summary['risk_level'] = pd.cut(
            station_summary['PM2.5'],
            bins=[0, 75, 85, max_pm25],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        st.dataframe(station_summary.sort_values(by='PM2.5', ascending=False).style.format({"PM2.5": "{:.2f}"}), use_container_width=True)

    st.divider()


    # --- KESIMPULAN ---
    st.subheader("📌 Kesimpulan")
    st.markdown("""
    Berdasarkan analisis data kualitas udara 2013-2017 pada 12 stasiun, ditemukan bahwa polusi PM2.5 
    menunjukkan pola musiman yang kuat, dengan puncak tertinggi pada bulan Desember/Januari dan titik terendah pada bulan Agustus.
    Secara spasial, terdapat ketimpangan signifikan di mana Stasiun Dongsi merupakan wilayah paling tercemar (42.7% waktu di atas ambang batas), sementara Stasiun Dingling relatif paling bersih. Secara keseluruhan, sekitar 39% waktu pengamatan menunjukkan kualitas udara kategori 'Tidak Sehat' hingga 'Sangat Tidak Sehat', 
    menegaskan perlunya kebijakan pengendalian emisi yang lebih ketat, khususnya di wilayah perkotaan pada musim dingin.""")

if __name__ == "__main__":
    main()
