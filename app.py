import streamlit as st
from ensemble_stack_v3 import final_prediction_pipeline
from utils import parse_manual_input
from visualizer import plot_confidences, plot_top_combinations
import pandas as pd

st.set_page_config(page_title="Prediksi Angka 4D SOTA", layout="wide")
st.title("🔢 Prediksi Angka 4D - SOTA Max Upgrade")

# Input data
st.markdown("### Masukkan Data Historis 4D (tanpa tanggal, hanya 4 digit per baris):")
manual_input = st.text_area("Contoh:\n1234\n4567\n6789", height=200)

if st.button("🔍 Prediksi Sekarang") and manual_input.strip():
    try:
        data = parse_manual_input(manual_input)
        hasil = final_prediction_pipeline(data)
    except Exception as e:
        st.error(f"❌ Gagal menjalankan pipeline: {e}")
        st.stop()

    st.success("✅ Prediksi berhasil diproses!")

    # Tampilkan hasil per posisi digit
    st.markdown("### 🎯 Top-3 Prediksi per Posisi:")
    posisi_label = ["Ribuan", "Ratusan", "Puluhan", "Satuan"]
    for i in range(4):
        st.markdown(f"**{posisi_label[i]}:**")
        top3 = hasil["top3_per_posisi"][i]
        confs = hasil["confidences"][i]
        plot_confidences(top3, confs, key=f"conf_{i}")

    # Tampilkan kombinasi 4D
    st.markdown("### 💡 Top-10 Kombinasi 4D Potensial:")
    df_top = pd.DataFrame(hasil["top10_kombinasi"], columns=["Angka 4D", "Skor"])
    st.dataframe(df_top, use_container_width=True)
    plot_top_combinations(df_top)

# Footer
st.markdown("---")
st.caption("🧠 Engine: Meta-LSTM + Transformer + Markov + Pola + Kalibrasi + Skoring + Logging")
