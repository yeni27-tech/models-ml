import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

API_URL = "http://localhost:5000"

st.set_page_config(page_title="Prediksi Laba/Rugi UMKM", layout="wide")

st.title("📊 Aplikasi Prediksi Laba/Rugi UMKM")

menu = st.sidebar.selectbox("Pilih Menu", ["Prediksi Laba/Rugi", "Riwayat Prediksi", "Visualisasi Data"])

if menu == "Prediksi Laba/Rugi":
    st.subheader("🧮 Form Input Prediksi")
    with st.form("prediction_form"):
        user_id = st.text_input("User ID", value="user_01")
        bidang_usaha = st.selectbox("Bidang Usaha", ["Jasa", "Manufaktur", "Perdagangan"])
        tahun = st.number_input("Tahun", min_value=2020, max_value=2030, value=2024)
        bulan = st.number_input("Bulan", min_value=1, max_value=12, value=4)
        pendapatan = st.number_input("Pendapatan", min_value=0.0, step=100000.0)
        beban_operasional = st.number_input("Beban Operasional", min_value=0.0, step=100000.0)
        pajak = st.number_input("Pajak", min_value=0.0, step=100000.0)
        model_choice = st.radio("Model yang digunakan", ["regresi", "time_series"])
        submitted = st.form_submit_button("Prediksi")

    if submitted:
        payload = {
            "User_ID": user_id,
            "Bidang_Usaha": bidang_usaha,
            "Tahun": int(tahun),
            "Bulan": int(bulan),
            "Pendapatan": pendapatan,
            "Beban_Operasional": beban_operasional,
            "Pajak": pajak,
            "model": model_choice
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Hasil Prediksi: {result['label']} ({result['prediction']:.2f})")
            else:
                st.error(f"Prediksi gagal. Status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Gagal menghubungkan ke server Flask. Pastikan Flask sudah berjalan di port 5000.")

elif menu == "Riwayat Prediksi":
    st.subheader("📄 Riwayat Prediksi")
    try:
        response = requests.get(f"{API_URL}/history")
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            if df.empty:
                st.info("Belum ada prediksi yang tersimpan.")
            else:
                df_sorted = df.sort_values(by="prediction", ascending=False)
                st.dataframe(df_sorted)
                csv = df_sorted.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", data=csv, file_name="riwayat_prediksi.csv", mime="text/csv")
        else:
            st.error("Gagal mengambil riwayat.")
    except requests.exceptions.ConnectionError:
        st.error("Tidak dapat terhubung ke server Flask.")

elif menu == "Visualisasi Data":
    st.subheader("📈 Visualisasi Prediksi")
    try:
        response = requests.get(f"{API_URL}/history")
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            if df.empty:
                st.info("Belum ada data untuk divisualisasikan.")
            else:
                bidang_selected = st.selectbox("Pilih Bidang Usaha", df['Bidang_Usaha'].unique())
                df_filtered = df[df['Bidang_Usaha'] == bidang_selected]

                fig, ax = plt.subplots()
                for user in df_filtered["User_ID"].unique():
                    user_df = df_filtered[df_filtered["User_ID"] == user]
                    ax.plot(user_df["Bulan"], user_df["prediction"], label=user)
                ax.set_title(f"Tren Prediksi Laba/Rugi - {bidang_selected}")
                ax.set_xlabel("Bulan")
                ax.set_ylabel("Nilai Prediksi")
                ax.legend()
                st.pyplot(fig)
        else:
            st.error("Gagal mengambil data.")
    except requests.exceptions.ConnectionError:
        st.error("Tidak dapat terhubung ke server Flask.")
