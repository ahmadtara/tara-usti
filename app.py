import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

st.title("Prediksi Status PO - My Republic 2024")

# Upload file
uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"])

if uploaded_file is not None:
    # Baca file
    df_raw = pd.read_excel(uploaded_file)

    # Simpan jumlah data awal
    initial_rows = len(df_raw)

    # Normalisasi nama kolom (hapus spasi, lowercase)
    df_raw.columns = df_raw.columns.str.strip()

    # Rename kolom agar sesuai format
    rename_map = {
        'Topology': 'topologi',
        'Vendor': 'vendor',
        'HP Cluster\n(SND Wajib Isi)': 'hp_cluster',
        'Status PO Cluster (SND Wajib Isi)': 'status_po'
    }
    df = df_raw.rename(columns=rename_map)

    # Pastikan kolom target ada
    required_cols = ['topologi', 'vendor', 'hp_cluster', 'status_po']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Kolom berikut tidak ditemukan: {missing_cols}")
        st.write("Kolom yang tersedia:", list(df.columns))
        st.stop()

    # Pilih hanya kolom yang diperlukan & drop NA
    df = df[required_cols].dropna()

    # Jumlah data setelah processing
    processed_rows = len(df)

    st.write(f"Jumlah data awal: **{initial_rows}**")
    st.write(f"Jumlah data setelah processing: **{processed_rows}**")

    # Bersihkan nilai teks
    df['status_po'] = df['status_po'].astype(str).str.strip().str.title()

    # Hitung distribusi status
    status_counts = df['status_po'].value_counts()

    # Plot distribusi
    fig, ax = plt.subplots()
    sns.barplot(x=status_counts.index, y=status_counts.values, ax=ax, palette="viridis")
    ax.set_xlabel("Status PO")
    ax.set_ylabel("Jumlah")
    ax.set_title("Jumlah Prediksi Tercapai vs Tidak")
    st.pyplot(fig)

    # One-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=False)

    target_col = 'status_po_Tercapai'
    if target_col not in df_encoded.columns:
        st.error(f"Kolom target '{target_col}' tidak ditemukan. Cek nilai di kolom status_po.")
        st.write("Nilai unik:", df['status_po'].unique())
        st.stop()

    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]

    # Split data train & test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)

    # Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    st.subheader("Hasil Akurasi Model")
    st.write(f"Decision Tree: **{acc_dt:.2%}**")
    st.write(f"Naive Bayes: **{acc_nb:.2%}**")

    st.subheader("Classification Report - Decision Tree")
    st.text(classification_report(y_test, y_pred_dt))

    st.subheader("Classification Report - Naive Bayes")
    st.text(classification_report(y_test, y_pred_nb))
