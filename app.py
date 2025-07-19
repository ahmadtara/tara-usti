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

st.title("Klasifikasi Data: Decision Tree vs Naive Bayes")

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
