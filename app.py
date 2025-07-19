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

st.title("Analisis Perbandingan Algoritma: C4.5 vs Naive Bayes Untuk Memprediksi Ketercapaian Target PO Dalam Membangun Project FTTH")

uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])

# Pilihan rasio split data
split_option = st.selectbox("Pilih rasio data latih vs uji", ("70:30", "80:20", "90:10"))
split_ratio = {"70:30": 0.3, "80:20": 0.2, "90:10": 0.1}[split_option]

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

    # Membersihkan dan mengonversi tipe data
    df = df.fillna("Unknown")
    for col in df.columns:
        df[col] = df[col].astype(str)

    # Encoding
    encoders = {col: LabelEncoder() for col in df.columns}
    for col, encoder in encoders.items():
        df[col] = encoder.fit_transform(df[col])

    # Heatmap korelasi
    st.subheader("Heatmap Korelasi")
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    # Split data
    X = df.drop('status_po', axis=1)
    y = df['status_po']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

    # Modeling Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    # Modeling Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)

    # Evaluasi akurasi
    acc_dt = accuracy_score(y_test, y_pred_dt)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    # Tabel perbandingan
    st.subheader("Tabel Perbandingan Akurasi")
    comparison_df = pd.DataFrame({
        "Algoritma": ["Decision Tree", "Naive Bayes"],
        "Akurasi": [acc_dt, acc_nb]
    })
    st.table(comparison_df)

    # Diagram batang perbandingan akurasi
    st.subheader("Diagram Batang Perbandingan Akurasi")
    fig_bar, ax_bar = plt.subplots()
    sns.barplot(x="Algoritma", y="Akurasi", data=comparison_df, palette="Set2", ax=ax_bar)
    ax_bar.set_ylim(0, 1)
    st.pyplot(fig_bar)

    # Matriks dan laporan klasifikasi
    st.subheader("Confusion Matrix & Classification Report")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Decision Tree")
        st.write(confusion_matrix(y_test, y_pred_dt))
        st.text(classification_report(y_test, y_pred_dt))

    with col2:
        st.write("Naive Bayes")
        st.write(confusion_matrix(y_test, y_pred_nb))
        st.text(classification_report(y_test, y_pred_nb))
