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

# === Title ===
st.title("📊 ML App: EDA + Decision Tree & Naive Bayes")
st.write("Upload file Excel → Pilih sheet → Lakukan EDA → Preprocessing → Training Model → Evaluasi")

# === Upload File ===
uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Pilih sheet
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        selected_sheet = st.selectbox("Pilih Sheet:", sheet_names)
        
        # Baca data
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        df = df.dropna()  # Drop baris kosong
        st.subheader("Preview Data")
        st.write(df.head())

        st.write(f"**Jumlah Data:** {df.shape[0]} baris, {df.shape[1]} kolom")

        # === EDA ===
        st.subheader("📈 Exploratory Data Analysis")

        st.write("### Statistik Deskriptif")
        st.write(df.describe(include="all"))

        # Korelasi
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 1:
            st.write("### Korelasi Heatmap")
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Tidak cukup kolom numerik untuk korelasi.")

        # Distribusi
        if len(numeric_cols) > 0:
            col = st.selectbox("Pilih kolom untuk distribusi:", numeric_cols)
            fig2, ax2 = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax2)
            st.pyplot(fig2)

        # === Preprocessing ===
        st.subheader("⚙️ Preprocessing Data")
        target_col = st.selectbox("Pilih kolom target:", df.columns)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Label Encoding target
        le = LabelEncoder()
        y = le.fit_transform(y)

        # One-hot encoding fitur kategorikal
        X_encoded = pd.get_dummies(X)

        # Pastikan semua numeric + isi NaN dengan 0
        X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Scaling
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # === Training Models ===
        st.subheader("🤖 Training Models")

        # Decision Tree
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)

        # Naive Bayes
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        y_pred_nb = nb.predict(X_test)

        # === Evaluation ===
        st.subheader("📊 Evaluasi Model")

        st.write("### Akurasi:")
        st.write(f"**Decision Tree:** {accuracy_score(y_test, y_pred_dt):.2f}")
        st.write(f"**Naive Bayes:** {accuracy_score(y_test, y_pred_nb):.2f}")

        # Classification Report
        st.write("### Classification Report (Decision Tree)")
        st.text(classification_report(y_test, y_pred_dt))

        st.write("### Classification Report (Naive Bayes)")
        st.text(classification_report(y_test, y_pred_nb))

        # Confusion Matrix
        st.write("### Confusion Matrix (Decision Tree)")
        cm_dt = confusion_matrix(y_test, y_pred_dt)
        fig3, ax3 = plt.subplots()
        sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues", ax=ax3)
        st.pyplot(fig3)

        st.write("### Confusion Matrix (Naive Bayes)")
        cm_nb = confusion_matrix(y_test, y_pred_nb)
        fig4, ax4 = plt.subplots()
        sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Greens", ax=ax4)
        st.pyplot(fig4)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

else:
    st.info("Silakan upload file Excel terlebih dahulu.")
