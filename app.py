import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)

st.title("Analisis Perbandingan Algoritma: C4.5 vs Naive Bayes Untuk Memprediksi Ketercapaian Target Po Dalam Membangun Project Ftth (Fiber To The Home)")

uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    df_raw = xls.parse('Sheet1')
    st.subheader("Data Awal")
    st.write(df_raw.head())

    # Tampilkan kolom asli
    st.subheader("Kolom Terdeteksi")
    st.write(df_raw.columns.tolist())

    # Bersihkan kolom dari spasi atau newline tersembunyi
    df_raw.columns = df_raw.columns.str.strip()

    # Mapping nama kolom asli ke nama baru
    column_map = {
        'Topology': 'topologi',
        'Vendor': 'vendor',
        'HP Cluster (SND Wajib Isi)': 'hp_cluster',
        'Status PO Cluster (SND Wajib Isi)': 'status_po'
    }

    df_renamed = df_raw.rename(columns=column_map)

    # Validasi apakah semua kolom target tersedia
    expected_cols = list(column_map.values())
    missing_cols = [col for col in expected_cols if col not in df_renamed.columns]
    if missing_cols:
        st.error(f"Kolom berikut tidak ditemukan setelah rename: {missing_cols}")
        st.stop()

    # Ambil hanya kolom yang dibutuhkan
    df = df_renamed[expected_cols].fillna("Unknown").astype(str)

    # Visualisasi distribusi target
    st.subheader("Distribusi Kelas Target (status_po)")
    fig_dist, ax_dist = plt.subplots()
    df['status_po'].value_counts().plot(kind='bar', ax=ax_dist, color='salmon')
    st.pyplot(fig_dist)

    # Visualisasi fitur kategorikal
    st.subheader("Distribusi Fitur Kategorikal")
    for col in ['topologi', 'vendor', 'hp_cluster']:
        fig_cat, ax_cat = plt.subplots()
        df[col].value_counts().plot(kind='bar', ax=ax_cat)
        ax_cat.set_title(f"Distribusi: {col}")
        st.pyplot(fig_cat)

    # Label encoding
    df['topologi'] = LabelEncoder().fit_transform(df['topologi'])
    df['vendor'] = LabelEncoder().fit_transform(df['vendor'])
    df['hp_cluster'] = LabelEncoder().fit_transform(df['hp_cluster'])
    df['status_po'] = LabelEncoder().fit_transform(df['status_po'])

    # Heatmap korelasi
    st.subheader("Heatmap Korelasi Fitur")
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    # Split dan evaluasi model
    X = df.drop('status_po', axis=1)
    y = df['status_po']
    results = []

    for ratio in [0.3, 0.2, 0.1]:  # 70:30, 80:20, 90:10
        split_str = f"{int((1 - ratio) * 100)}:{int(ratio * 100)}"
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)

        models = {
            "Decision Tree": DecisionTreeClassifier(),
            "Naive Bayes": GaussianNB()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append({
                "Model": name,
                "Split": split_str,
                "Akurasi": accuracy_score(y_test, y_pred),
                "Presisi": precision_score(y_test, y_pred, average='macro', zero_division=0),
                "Recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, average='macro', zero_division=0)
            })

    result_df = pd.DataFrame(results)
    st.subheader("Tabel Evaluasi Akurasi, Presisi, Recall, F1")
    st.dataframe(result_df)

    # Grafik akurasi
    st.subheader("Grafik Perbandingan Akurasi")
    fig, ax = plt.subplots()
    for model in result_df['Model'].unique():
        data = result_df[result_df['Model'] == model]
        ax.plot(data['Split'], data['Akurasi'], marker='o', label=model)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Akurasi")
    ax.set_title("Perbandingan Akurasi Berdasarkan Split Data")
    ax.legend()
    st.pyplot(fig)

    # ROC Curve dengan 80:20 split
    st.subheader("ROC Curve (Split 80:20)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_bin = label_binarize(y_test, classes=np.unique(y))

    fig_roc, ax_roc = plt.subplots()
    for name, model in models.items():
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)
