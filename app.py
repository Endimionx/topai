import streamlit as st
import pandas as pd
from models import full_prediction_pipeline
from utils import preprocess_input, show_prediction_results
from simulator import simulate_prediction_accuracy
from visualizer import plot_confidences

st.set_page_config(layout="centered", page_title="Prediksi Angka 4D - SOTA")

st.title("🔢 Prediksi Angka 4D - SOTA Full System")
st.markdown("Masukkan data historis 4D (tanpa tanggal), satu angka per baris. Contoh:\n\n```\n1234\n5678\n9012\n...```")

text_input = st.text_area("Input Data Historis (minimal 50 angka):", height=300)

if text_input:
    data = preprocess_input(text_input)
    if len(data) < 50:
        st.warning("Minimal 50 data 4D diperlukan.")
    else:
        with st.spinner("Memproses data dan melakukan prediksi..."):
            predictions, confidences = full_prediction_pipeline(data)

        show_prediction_results(predictions, confidences)
        plot_confidences(confidences)

        acc = simulate_prediction_accuracy(data, full_prediction_pipeline)
        if acc:
            top1, top3 = acc
            st.subheader("📌 Simulasi Akurasi Real")
            st.markdown(f"🎯 **Top-1 Akurasi**: {top1:.2%}")
            st.markdown(f"🎯 **Top-3 Akurasi**: {top3:.2%}")
