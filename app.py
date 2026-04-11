import zipfile
from pathlib import Path
import gdown
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Konfigurasi Halaman
st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide",
)

sns.set_style("whitegrid")

# Konstanta
FILE_ID = "1RhU3gJlkteaAQfyn9XOVAz7a5o1-etgr"
ZIP_PATH = Path("air_quality.zip")
DATA_DIR = Path("data")

@st.cache_data(show_spinner=True)
def load_data():
    # 1. Pastikan folder data ada
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(exist_ok=True)

    # 2. Cek apakah ada file CSV di dalam folder data (termasuk subfolder)
    csv_files = sorted(DATA_DIR.rglob("*.csv"))

    # 3. Jika tidak ada CSV, download dan ekstrak
    if not csv_files:
        if not ZIP_PATH.exists():
            gdown.download(
                f"https://drive.google.com/uc?id={FILE_ID}",
                str(ZIP_PATH),
                quiet=False,
            )
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        # Cari ulang list file setelah diekstrak
        csv_files = sorted(DATA_DIR.rglob("*.csv"))

    if not csv_files:
        st.error("Gagal menemukan file CSV. Pastikan struktur folder di GitHub sudah benar.")
        st.stop()

    df_list = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            # NORMALISASI: Kecilkan semua nama kolom (mencegah KeyError 'station' vs 'Station')
            df.columns = [c.lower() for c in df.columns]

            # Jika kolom 'station' tidak ada, ambil dari nama file
            if "station" not in df.columns or df["station"].isna().all():
                parts = file_path.stem.split("_")
                station_name = parts[2] if len(parts) > 2 else file_path.stem
                df["station"] = station_name

            df_list.append(df)
        except Exception as e:
            st.warning(f"Gagal membaca {file_path.name}: {e}")

    # Gabungkan semua data
    full_df = pd.concat(df_list, ignore_index=True)

    # Buat kolom datetime
    full_df["datetime"] = pd.to_datetime(
        full_df[["year", "month", "day", "hour"]],
        errors="coerce"
    )

    # Bersihkan data
    full_df = full_df.dropna(subset=["datetime"])
    full_df = full_df.sort_values(["station", "datetime"]).reset_index(drop=True)

    # Isi missing value (ffill & bfill) per stasiun
    # Gunakan reindex/logic yang lebih aman untuk pandas versi baru
    full_df = full_df.groupby("station", group_keys=False).apply(lambda x: x.ffill().bfill()).reset_index(drop=True)

    return full_df

def filter_data(df):
    stations = sorted(df["station"].unique().tolist())
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

# --- Fungsi Plotting ---
def plot_monthly_pm25(df):
    monthly = df.groupby(df["datetime"].dt.month)["pm2.5"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly.values, marker="o", color="teal")
    ax.set_title("Rata-rata Konsentrasi PM2.5 per Bulan")
    ax.set_xticks(range(1, 13))
    st.pyplot(fig)

def plot_hourly_pm25(df):
    hourly = df.groupby(df["datetime"].dt.hour)["pm2.5"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourly.index, hourly.values, marker="o", color="orange")
    ax.set_title("Rata-rata Konsentrasi PM2.5 per Jam")
    st.pyplot(fig)

def plot_daily_pm25(df):
    daily_avg = df.set_index("datetime").resample("D")["pm2.5"].mean()
    rolling = daily_avg.rolling(window=7).mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily_avg.index, daily_avg.values, alpha=0.3, label="Harian")
    ax.plot(rolling.index, rolling.values, color="red", label="Rata-rata 7 Hari")
    ax.set_title("Tren Harian PM2.5")
    ax.legend()
    st.pyplot(fig)

def plot_station_pm25(df):
    station_avg = df.groupby("station")["pm2.5"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=station_avg.index, y=station_avg.values, ax=ax, palette="viridis")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    return station_avg

# --- Main App ---
def main():
    st.title("🌫️ Dashboard Analisis Kualitas Udara")
    
    df = load_data()

    # Sidebar
    st.sidebar.header("Filter")
    filtered_df, sel_stat, start, end = filter_data(df)

    if filtered_df.empty:
        st.warning("Data kosong untuk filter ini.")
        return

    # Metrics
    st.subheader("Ringkasan Data")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Baris", f"{len(filtered_df):,}")
    m2.metric("Jumlah Stasiun", filtered_df["station"].nunique())
    m3.metric("Rerata PM2.5", f"{filtered_df['pm2.5'].mean():.2f}")

    # Visualisasi
    st.subheader("Pola Perubahan Waktu")
    c1, c2 = st.columns(2)
    with c1: plot_monthly_pm25(filtered_df)
    with c2: plot_hourly_pm25(filtered_df)
    
    plot_daily_pm25(filtered_df)

    st.subheader("Analisis per Stasiun")
    station_avg = plot_station_pm25(filtered_df)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Top 3 Stasiun Polusi Tertinggi**")
        st.table(station_avg.head(3))
    with col_b:
        st.write("**Top 3 Stasiun Polusi Terendah**")
        st.table(station_avg.tail(3))

    st.subheader("Data Sampel")
    st.dataframe(filtered_df.sample(min(10, len(filtered_df))), use_container_width=True)

if __name__ == "__main__":
    main()
