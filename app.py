import streamlit as st
from ensemble_stack_v3 import final_prediction_pipeline
from utils import preprocess_input
from visualizer import plot_confidences

st.set_page_config(layout="centered", page_title="Prediksi Angka 4D - Ultimate SOTA")

st.title("🔮 Prediksi Angka 4D - Meta + Pattern Aware")
st.markdown("Masukkan angka 4D tanpa tanggal. Contoh:\n\n```\n1234\n4567\n8901\n...```")

text_input = st.text_area("📥 Input Data Historis 4D", height=300)

if text_input:
    data = preprocess_input(text_input)
    if len(data) < 50:
        st.warning("Minimal 50 angka 4D diperlukan.")
    else:
        with st.spinner("🔁 Memproses dengan ensemble + meta-learning..."):
            hasil = final_prediction_pipeline(data)

        top3 = hasil["top3_per_posisi"]
        confs = hasil["confidences"]
        kombinasi = hasil["top10_kombinasi"]

        posisi = ['Ribuan', 'Ratusan', 'Puluhan', 'Satuan']
        st.subheader("🎯 Prediksi Top-3 per Posisi (Meta-Voting)")
        for i, pos in enumerate(posisi):
            st.markdown(f"**{pos}**:")
            for j in range(3):
                st.write(f"{j+1}. Angka: {top3[i][j]} | Confidence: {confs[i][j]:.2%}")

        plot_confidences(confs)

        st.subheader("💡 Kombinasi 4D Terbaik Berdasarkan Pola")
        for comb, score in kombinasi:
            st.write(f"**{''.join(map(str, comb))}** | Skor Pola: {score}")
