import zipfile
from pathlib import Path

import gdown
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide",
)

sns.set_style("whitegrid")

# Pengaturan Global
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

    # 3. Kalau CSV nggak ada, baru coba download & ekstrak
    if not csv_files:
        if not ZIP_PATH.exists():
            try:
                gdown.download(
                    f"https://drive.google.com/uc?id={FILE_ID}",
                    str(ZIP_PATH),
                    quiet=False,
                )
            except Exception as e:
                st.error(f"Gagal download data: {e}")
        
        if ZIP_PATH.exists():
            with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall(DATA_DIR)
            # Update daftar file setelah diekstrak
            csv_files = sorted(DATA_DIR.rglob("*.csv"))

    # 4. Final check kalau masih nggak ada file
    if not csv_files:
        st.error(f"Waduh! File CSV tidak ditemukan di dalam folder {DATA_DIR}. Pastikan kamu sudah mengupload data CSV ke GitHub.")
        st.stop() # Berhenti di sini biar nggak error KeyError

    df_list = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)

        # Cek dan buat kolom 'station' jika tidak ada
        if "station" not in df.columns or df["station"].isna().all():
            parts = file_path.stem.split("_")
            station_name = parts[2] if len(parts) > 2 else file_path.stem
            df["station"] = station_name
        
        df_list.append(df)

    # Gabungkan semua data
    df = pd.concat(df_list, ignore_index=True)

    # Buat kolom datetime
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]],
        errors="coerce"
    )

    # Hapus baris datetime gagal
    df = df.dropna(subset=["datetime"])

    # Urutkan
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)

    # Isi missing value per station
    df = (
        df.groupby("station", group_keys=False)
        .apply(lambda x: x.ffill().bfill())
        .reset_index(drop=True)
    )

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


def plot_monthly_pm25(df):
    monthly = (
        df.groupby(df["datetime"].dt.month)["PM2.5"]
        .mean()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly.values, marker="o")
    ax.set_title("Rata-rata Konsentrasi PM2.5 per Bulan")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Rata-rata PM2.5")
    ax.set_xticks(monthly.index)
    st.pyplot(fig)


def plot_hourly_pm25(df):
    hourly = df.groupby(df["datetime"].dt.hour)["PM2.5"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourly.index, hourly.values, marker="o")
    ax.set_title("Rata-rata Konsentrasi PM2.5 per Jam")
    ax.set_xlabel("Jam")
    ax.set_ylabel("Rata-rata PM2.5")
    ax.set_xticks(hourly.index)
    st.pyplot(fig)


def plot_daily_pm25(df):
    daily_pm25 = (
        df.set_index("datetime")
        .groupby("station")["PM2.5"]
        .resample("D")
        .mean()
        .reset_index()
    )

    daily_pm25_avg = daily_pm25.groupby("datetime")["PM2.5"].mean()
    rolling_pm25 = daily_pm25_avg.rolling(window=7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily_pm25_avg.index, daily_pm25_avg.values, alpha=0.4, label="Harian")
    ax.plot(rolling_pm25.index, rolling_pm25.values, linewidth=2, label="Rata-rata 7 Hari")
    ax.set_title("Rata-rata Konsentrasi PM2.5 Harian")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Rata-rata PM2.5")
    ax.legend()
    st.pyplot(fig)


def plot_station_pm25(df):
    station_avg = df.groupby("station")["PM2.5"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=station_avg.index, y=station_avg.values, ax=ax)
    ax.set_title("Rata-rata Konsentrasi PM2.5 per Stasiun")
    ax.set_xlabel("Stasiun")
    ax.set_ylabel("Rata-rata PM2.5")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

    return station_avg


def main():
    st.title("🌫️ Dashboard Analisis Kualitas Udara")
    st.markdown(
        """
        Dashboard ini merupakan versi Streamlit dari notebook analisis data Air Quality Dataset.

        Fokus analisis:
        1. Pola perubahan konsentrasi PM2.5 berdasarkan waktu pengamatan.
        2. Stasiun dengan rata-rata konsentrasi PM2.5 tertinggi dan terendah.
        """
    )

    st.sidebar.header("Filter")

    df = load_data()

    # Debug info
    with st.expander("Cek data yang terbaca"):
        st.write("Jumlah file/stasiun terbaca:", df["station"].nunique())
        st.write("Daftar station:", sorted(df["station"].dropna().unique().tolist()))

    filtered_df, selected_stations, start_date, end_date = filter_data(df)

    if filtered_df.empty:
        st.warning("Tidak ada data yang sesuai filter.")
        return

    st.subheader("Ringkasan Data")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah baris", f"{len(filtered_df):,}")
    c2.metric("Jumlah stasiun", filtered_df["station"].nunique())
    c3.metric("Rata-rata PM2.5", f"{filtered_df['PM2.5'].mean():.2f}")
    c4.metric("Maks PM2.5", f"{filtered_df['PM2.5'].max():.2f}")

    st.caption(
        f"Filter aktif: {len(selected_stations)} stasiun, periode {start_date} s.d. {end_date}"
    )

    st.subheader("Pertanyaan 1: Bagaimana pola perubahan konsentrasi PM2.5 berdasarkan waktu pengamatan?")
    col1, col2 = st.columns(2)
    with col1:
        plot_monthly_pm25(filtered_df)
    with col2:
        plot_hourly_pm25(filtered_df)

    plot_daily_pm25(filtered_df)

    st.subheader("Pertanyaan 2: Stasiun mana yang memiliki rata-rata konsentrasi PM2.5 tertinggi dan terendah?")
    station_avg = plot_station_pm25(filtered_df)

    top3 = station_avg.head(3).rename("PM2.5").reset_index()
    bottom3 = station_avg.tail(3).rename("PM2.5").reset_index()

    a, b = st.columns(2)
    with a:
        st.markdown("**3 Stasiun dengan rata-rata PM2.5 tertinggi**")
        st.dataframe(top3, use_container_width=True)
    with b:
        st.markdown("**3 Stasiun dengan rata-rata PM2.5 terendah**")
        st.dataframe(bottom3, use_container_width=True)

    st.subheader("Data Sampel")
    sample_cols = ["datetime", "station", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]

    # acak supaya tidak cuma station pertama
    sample_df = (
        filtered_df[sample_cols]
        .sample(n=min(20, len(filtered_df)), random_state=42)
        .sort_values(["datetime", "station"])
        .reset_index(drop=True)
    )

    st.dataframe(sample_df, use_container_width=True)

    st.subheader("Kesimpulan")
    st.markdown(
        """
        - Konsentrasi PM2.5 menunjukkan pola perubahan menurut bulan, jam, dan tren harian.
        - Terdapat perbedaan rata-rata PM2.5 antar stasiun, menandakan distribusi polusi tidak merata.
        - Dashboard ini mendukung eksplorasi interaktif berdasarkan filter stasiun dan rentang tanggal.
        """
    )


if __name__ == "__main__":
    main()
