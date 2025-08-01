import streamlit as st

def preprocess_input(text):
    lines = text.strip().split("\n")
    valid = [line.strip() for line in lines if line.strip().isdigit() and len(line.strip()) == 4]
    return [list(map(int, list(item))) for item in valid]

def show_prediction_results(predictions, confidences):
    posisi = ['Ribuan', 'Ratusan', 'Puluhan', 'Satuan']
    st.subheader("🎯 Hasil Prediksi Top-3 Tiap Posisi dengan Confidence")
    for idx, pos in enumerate(posisi):
        top3 = predictions[idx]
        conf = confidences[idx]
        st.markdown(f"**{pos}**:")
        for i in range(3):
            st.write(f"{i+1}. Angka: {top3[i]} | Confidence: {conf[i]:.2%}")
