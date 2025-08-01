import matplotlib.pyplot as plt
import streamlit as st

def plot_confidences(digits, confidences, key=None):
    """
    Menampilkan bar chart confidence top-3 digit prediksi
    """
    fig, ax = plt.subplots()
    bars = ax.bar([str(d) for d in digits], confidences, color="skyblue")
    ax.set_ylim(0, 1)
    ax.set_title("Confidence Top-3")
    for bar, conf in zip(bars, confidences):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{conf:.2f}", ha='center', va='bottom')
    st.pyplot(fig, clear_figure=True)

def plot_top_combinations(df_top):
    """
    Visualisasi kombinasi 4D terbaik (misalnya dari hasil pattern-aware + skoring)
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    angka = df_top["Angka 4D"]
    skor = df_top["Skor"]
    bars = ax.bar(angka, skor, color="lightgreen")
    ax.set_title("Top-10 Kombinasi 4D Potensial")
    ax.set_ylabel("Skor")
    ax.set_xticklabels(angka, rotation=45)
    for bar, s in zip(bars, skor):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{s:.2f}", ha='center', va='bottom')
    st.pyplot(fig, clear_figure=True)
