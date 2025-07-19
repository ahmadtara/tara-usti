import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Analisis Perbandingan Algoritma: C4.5 vs Naive Bayes Untuk Memprediksi Ketercapaian Target Po Dalam Membangun Project FTTH")

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
    for col in df.columns:
        df[col] = df[col].astype(str)

    # Encoding
    label_encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Visualisasi Heatmap
    st.subheader("Heatmap Korelasi Fitur")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Split Ratio Option
    st.subheader("Pilih Rasio Split Data")
    split_option = st.selectbox("Rasio Split Data", ("70:30", "80:20", "90:10"))
    test_size = {"70:30": 0.3, "80:20": 0.2, "90:10": 0.1}[split_option]

    # Split
    X = df.drop('status_po', axis=1)
    y = df['status_po']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Modeling
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)

    # Evaluation
    st.subheader("Akurasi")
    st.write(f"Decision Tree Accuracy ({split_option}):", accuracy_score(y_test, y_pred_dt))
    st.write(f"Naive Bayes Accuracy ({split_option}):", accuracy_score(y_test, y_pred_nb))

    st.subheader("Confusion Matrix: Decision Tree")
    st.write(confusion_matrix(y_test, y_pred_dt))

    st.subheader("Confusion Matrix: Naive Bayes")
    st.write(confusion_matrix(y_test, y_pred_nb))

    st.subheader("Classification Report: Decision Tree")
    st.text(classification_report(y_test, y_pred_dt))

    st.subheader("Classification Report: Naive Bayes")
    st.text(classification_report(y_test, y_pred_nb))

    # ROC Curve (untuk klasifikasi biner saja)
    if len(np.unique(y)) == 2:
        st.subheader("ROC Curve")

        y_test_bin = label_binarize(y_test, classes=[0, 1]).ravel()

        y_score_dt = dt.predict_proba(X_test)[:, 1]
        y_score_nb = nb.predict_proba(X_test)[:, 1]

        fpr_dt, tpr_dt, _ = roc_curve(y_test_bin, y_score_dt)
        fpr_nb, tpr_nb, _ = roc_curve(y_test_bin, y_score_nb)

        auc_dt = auc(fpr_dt, tpr_dt)
        auc_nb = auc(fpr_nb, tpr_nb)

        fig_roc, ax = plt.subplots()
        ax.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {auc_dt:.2f})")
        ax.plot(fpr_nb, tpr_nb, label=f"Naive Bayes (AUC = {auc_nb:.2f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig_roc)
    else:
        st.warning("ROC Curve hanya dapat ditampilkan jika label hanya terdiri dari 2 kelas (biner).")
