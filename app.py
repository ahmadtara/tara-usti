import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("Analisis Klasifikasi Data Site Cluster")

uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    df_raw = xls.parse('Sheet1')

    df = df_raw.rename(columns={
        'Topology': 'topologi',
        'Vendor': 'vendor',
        'HP Cluster\n(SND Wajib Isi)': 'hp_cluster',
        'Status PO Cluster (SND Wajib Isi)': 'status_po'
    })
    df = df[['topologi', 'vendor', 'hp_cluster', 'status_po']]
    st.subheader("Data Setelah Seleksi dan Rename")
    st.write(df.head())

    # Label Encoding dan Scaling
    st.subheader("Pra-pemrosesan Data")
    label = LabelEncoder()
    df['topologi'] = label.fit_transform(df['topologi'])
    df['vendor'] = label.fit_transform(df['vendor'])
    df['hp_cluster'] = label.fit_transform(df['hp_cluster'])
    df['status_po'] = label.fit_transform(df['status_po'])

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    st.write("Data Setelah Scaling")
    st.write(df_scaled.head())

    # Split data
    X = df_scaled.drop('status_po', axis=1)
    y = df_scaled['status_po']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Decision Tree
    st.subheader("Model Decision Tree")
    model_dt = DecisionTreeClassifier()
    model_dt.fit(X_train, y_train)
    y_pred_dt = model_dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    st.write("Akurasi:", acc_dt)
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred_dt))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred_dt))

    # Model Naive Bayes
    st.subheader("Model Naive Bayes")
    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    y_pred_nb = model_nb.predict(X_test)
    acc_nb =_
