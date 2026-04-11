import zipfile
from pathlib import Path
import gdown
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Air Quality Dashboard", page_icon="🌫️", layout="wide")
sns.set_style("whitegrid")

FILE_ID = "1RhU3gJlkteaAQfyn9XOVAz7a5o1-etgr"
ZIP_PATH = Path("air_quality.zip")
DATA_DIR = Path("data")

@st.cache_data(show_spinner=True)
def load_data():
    if not DATA_DIR.exists(): DATA_DIR.mkdir(exist_ok=True)
    
    # Cari CSV di mana aja (biar gak error path)
    csv_files = sorted(DATA_DIR.rglob("*.csv"))

    if not csv_files:
        if not ZIP_PATH.exists():
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", str(ZIP_PATH), quiet=False)
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        csv_files = sorted(DATA_DIR.rglob("*.csv"))

    if not csv_files:
        st.error("File CSV tidak ditemukan!"); st.stop()

    df_list = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        if "station" not in df.columns or df["station"].isna().all():
            df["station"] = file_path.stem.split("_")[2] if "_" in file_path.stem else file_path.stem
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values(["station", "datetime"]).reset_index(drop=True)
    df = df.groupby("station", group_keys=False).apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
    return df

def filter_data(df):
    stations = sorted(df["station"].unique().tolist())
    sel_stations = st.sidebar.multiselect("Pilih stasiun", options=stations, default=stations)
    min_d, max_d = df["datetime"].min().date(), df["datetime"].max().date()
    sel_dates = st.sidebar.date_input("Rentang tanggal", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    
    start_d, end_d = (sel_dates[0], sel_dates[1]) if isinstance(sel_dates, tuple) and len(sel_dates)==2 else (min_d, max_d)
    filtered = df[(df["station"].isin(sel_stations)) & (df["datetime"].dt.date >= start_d) & (df["datetime"].dt.date <= end_d)].copy()
    return filtered, sel_stations, start_d, end_d

def main():
    st.title("🌫️ Dashboard Analisis Kualitas Udara")
    df = load_data()
    
    with st.expander("Cek data"):
        st.write("Stasiun:", df["station"].unique())

    f_df, sel_stat, sd, ed = filter_data(df)
    if f_df.empty: st.warning("Data kosong."); return

    st.subheader("Ringkasan Data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baris", f"{len(f_df):,}")
    c2.metric("Stasiun", f_df["station"].nunique())
    c3.metric("Rata-rata PM2.5", f"{f_df['PM2.5'].mean():.2f}")

    col1, col2 = st.columns(2)
    with col1:
        mon = f_df.groupby(f_df["datetime"].dt.month)["PM2.5"].mean()
        fig, ax = plt.subplots(); ax.plot(mon.index, mon.values, marker="o"); ax.set_title("Bulanan"); st.pyplot(fig)
    with col2:
        hr = f_df.groupby(f_df["datetime"].dt.hour)["PM2.5"].mean()
        fig, ax = plt.subplots(); ax.plot(hr.index, hr.values, marker="o"); ax.set_title("Jam"); st.pyplot(fig)

    st.subheader("PM2.5 per Stasiun")
    st_avg = f_df.groupby("station")["PM2.5"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(); sns.barplot(x=st_avg.index, y=st_avg.values, ax=ax); plt.xticks(rotation=45); st.pyplot(fig)

    st.subheader("Data Sampel")
    st.dataframe(f_df[["datetime", "station", "PM2.5"]].sample(min(10, len(f_df))), use_container_width=True)

if __name__ == "__main__":
    main()
