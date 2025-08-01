import streamlit as st
import pandas as pd
from models import full_prediction_pipeline
from utils import preprocess_input, show_prediction_results

st.set_page_config(layout="centered", page_title="Prediksi Angka 4D - SOTA")

st.title("🔢 Prediksi Angka 4D - SOTA Full System")
st.markdown("Masukkan data historis 4D (tanpa tanggal) di bawah ini. Contoh format:1234\n2345\n3456")

text_input = st.text_area("Input Data Historis (minimal 50 angka):", height=300)

if text_input:
    data = preprocess_input(text_input)
    if len(data) < 50:
        st.warning("Minimal 50 data 4D diperlukan.")
    else:
        with st.spinner("Memproses data dan melakukan prediksi..."):
            predictions, confidences = full_prediction_pipeline(data)
        show_prediction_results(predictions, confidences)
