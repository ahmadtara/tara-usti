import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("Analisis Perbandingan Algoritma: C4.5 vs Naive Bayes Untuk Memprediksi Ketercapaian Target Po Dalam Membangun Project Ftth (Fiber To The Home)")

uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])

if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    df_raw = xls.parse('Sheet1')
    st.subheader("Data Awal")
    st.write(df_raw.head())

    # Rename kolom
    df = df_raw.rename(columns={
        'Topology': 'topologi',
        'Vendor': 'vendor',
        'HP Cluster\n(SND Wajib Isi)': 'hp_cluster',
        'Status PO Cluster (SND Wajib Isi)': 'status_po'
    })

    df = df[['topologi', 'vendor', 'hp_cluster', 'status_po']]
    st.subheader("Data yang Digunakan")
    st.write(df.head())

    # Bersihkan dan ubah tipe data
    df = df.fillna("Unknown")
    df['topologi'] = df['topologi'].astype(str)
    df['vendor'] = df['vendor'].astype(str)
    df['hp_cluster'] = df['hp_cluster'].astype(str)
    df['status_po'] = df['status_po'].astype(str)

    # Encoding
    le_topologi = LabelEncoder()
    le_vendor = LabelEncoder()
    le_hp_cluster = LabelEncoder()
    le_status_po = LabelEncoder()

    df['topologi'] = le_topologi.fit_transform(df['topologi'])
    df['vendor'] = le_vendor.fit_transform(df['vendor'])
    df['hp_cluster'] = le_hp_cluster.fit_transform(df['hp_cluster'])
    df['status_po'] = le_status_po.fit_transform(df['status_po'])

    # Split
    X = df.drop('status_po', axis=1)
    y = df['status_po']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modeling
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)

    acc_dt = accuracy_score(y_test, y_pred_dt)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    # Tampilkan Akurasi
    st.subheader("Akurasi Model")
    st.write("Decision Tree Accuracy:", acc_dt)
    st.write("Naive Bayes Accuracy:", acc_nb)

    # Tabel Perbandingan Akurasi
    st.subheader("Tabel Perbandingan Akurasi")
    acc_df = pd.DataFrame({
        "Model": ["Decision Tree", "Naive Bayes"],
        "Akurasi": [acc_dt, acc_nb]
    })
    st.dataframe(acc_df)

    # Grafik Perbandingan Akurasi
    st.subheader("Grafik Perbandingan Akurasi")
    fig, ax = plt.subplots()
    ax.bar(acc_df["Model"], acc_df["Akurasi"], color=["skyblue", "lightgreen"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Akurasi")
    ax.set_title("Perbandingan Akurasi Model")
    for i, v in enumerate(acc_df["Akurasi"]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
    st.pyplot(fig)

    # Confusion Matrix dan Classification Report
    st.subheader("Confusion Matrix: Decision Tree")
    st.write(confusion_matrix(y_test, y_pred_dt))

    st.subheader("Confusion Matrix: Naive Bayes")
    st.write(confusion_matrix(y_test, y_pred_nb))

    st.subheader("Classification Report: Decision Tree")
    st.text(classification_report(y_test, y_pred_dt))

    st.subheader("Classification Report: Naive Bayes")
    st.text(classification_report(y_test, y_pred_nb))
