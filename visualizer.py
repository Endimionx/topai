import matplotlib.pyplot as plt
import streamlit as st

def plot_confidences(confidences):
    posisi = ['Ribuan', 'Ratusan', 'Puluhan', 'Satuan']
    st.subheader("📊 Confidence Score Visualisasi")

    for i, conf in enumerate(confidences):
        fig, ax = plt.subplots()
        bars = ax.bar(range(1, 4), conf, tick_label=['Top1', 'Top2', 'Top3'])
        ax.set_ylim(0, 1)
        ax.set_title(f"Confidence - {posisi[i]}")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2%}", ha='center', va='bottom')
        st.pyplot(fig)
