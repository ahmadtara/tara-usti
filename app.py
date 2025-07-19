import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)

st.title("Analisis Perbandingan Algoritma: C4.5 vs Naive Bayes Untuk Memprediksi Ketercapaian Target Po Dalam Membangun Project Ftth (Fiber To The Home)")

uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])

# Pilihan split data
split_option = st.selectbox("Pilih rasio data latih vs uji", ("70:30", "80:20", "90:10"))
split_ratio = {"70:30": 0.3, "80:20": 0.2, "90:10": 0.1}[split_option]

if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    df_raw = xls.parse('Sheet1')
    st.subheader("Data Awal")
    st.write(df_raw.head())

    df = df_raw.rename(columns={
        'Topology': 'topologi',
        'Vendor': 'vendor',
        'HP Cluster\n(SND Wajib Isi)': 'hp_cluster',
        'Status PO Cluster (SND Wajib Isi)': 'status_po'
    })

    df = df[['topologi', 'vendor', 'hp_cluster', 'status_po']]
    st.subheader("Data yang Digunakan")
    st.write(df.head())

    df = df.fillna("Unknown")
    for col in df.columns:
        df[col] = df[col].astype(str)

    # Encoding
    encoders = {col: LabelEncoder() for col in df.columns}
    for col, encoder in encoders.items():
        df[col] = encoder.fit_transform(df[col])

    # Visualisasi distribusi data
    st.subheader("Distribusi Setiap Fitur")
    for col in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    # Heatmap korelasi
    st.subheader("Heatmap Korelasi")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Split data
    X = df.drop('status_po', axis=1)
    y = df['status_po']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

    # Modeling
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)

    # Evaluation
    st.subheader("Akurasi")
    st.write("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
    st.write("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

    st.subheader("Confusion Matrix: Decision Tree")
    st.write(confusion_matrix(y_test, y_pred_dt))

    st.subheader("Confusion Matrix: Naive Bayes")
    st.write(confusion_matrix(y_test, y_pred_nb))

    st.subheader("Classification Report: Decision Tree")
    st.text(classification_report(y_test, y_pred_dt))

    st.subheader("Classification Report: Naive Bayes")
    st.text(classification_report(y_test, y_pred_nb))

    # ROC Curve
    if len(np.unique(y)) == 2:
        st.subheader("ROC Curve")
        fpr_dt, tpr_dt, _ = roc_curve(y_test, dt.predict_proba(X_test)[:, 1])
        fpr_nb, tpr_nb, _ = roc_curve(y_test, nb.predict_proba(X_test)[:, 1])
        auc_dt = auc(fpr_dt, tpr_dt)
        auc_nb = auc(fpr_nb, tpr_nb)

        fig, ax = plt.subplots()
        ax.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})')
        ax.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {auc_nb:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("ROC Curve hanya tersedia untuk data klasifikasi biner.")
