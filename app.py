import streamlit as st
from ensemble_stack_v3 import final_prediction_pipeline
from drift_monitor import calculate_accuracy_per_position

st.set_page_config(page_title="Prediksi 4D - SOTA Max", layout="centered")
st.title("🔮 Prediksi Angka 4D - SOTA Max Upgrade")

st.markdown("Masukkan data historis angka 4D (tanpa tanggal), satu per baris:")
st.code("1234\n4520\n0987")

# Input TextArea
raw_input = st.text_area("Data Historis 4D", height=300)
data = []

# Validasi Input
if raw_input:
    try:
        for line in raw_input.strip().split("\n"):
            line = line.strip()
            assert len(line) == 4
            digits = [int(ch) for ch in line]
            assert all(0 <= d <= 9 for d in digits)
            data.append(digits)
    except:
        st.error("⚠️ Format salah. Harus 4 digit angka per baris, contoh: 1234")

# Proses Prediksi
if len(data) >= 20:
    with st.spinner("🚀 Memproses prediksi dengan ensemble SOTA..."):
        result = final_prediction_pipeline(data)
        predictions = result["top3_per_posisi"]
        confidences = result["confidences"]
        top10_kombinasi = result["top10_kombinasi"]

    st.success("✅ Prediksi berhasil dihitung.")

    # Tampilkan Hasil Per Posisi
    st.subheader("📊 Top-3 Digit per Posisi")
    posisi = ["Ribuan", "Ratusan", "Puluhan", "Satuan"]
    for i in range(4):
        top3 = predictions[i]
        conf = ["{:.2f}".format(c) for c in confidences[i]]
        st.write(f"**{posisi[i]}** → {top3} (Conf: {conf})")

    # Kombinasi Final
    st.subheader("🔢 Top-10 Kombinasi 4D Terbaik")
    for i, komb in enumerate(top10_kombinasi, 1):
        st.write(f"{i}. {' '.join(map(str, komb))}")

    # Live Accuracy
    st.subheader("📈 Akurasi Real-Time per Posisi")
    acc = calculate_accuracy_per_position()
    for i in range(4):
        st.write(f"{posisi[i]}: **{acc.get(i, 0)}%**")
else:
    st.info("ℹ️ Masukkan minimal 20 baris data historis 4D untuk menjalankan prediksi.")
