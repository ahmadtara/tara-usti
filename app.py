import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score
)

st.title("📊 Perbandingan C4.5 (Decision Tree) vs Naive Bayes Untuk Prediksi Target PO - FTTH Project")

uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])

if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    df_raw = xls.parse('Sheet1')

    st.subheader("🗂️ Data Mentah")
    st.write(df_raw.head())

    # Bersihkan nama kolom dari spasi dan newline
    df_raw.columns = df_raw.columns.str.strip().str.replace('\n', ' ', regex=False)

    # Mapping kolom
    column_map = {
        'Topology': 'topologi',
        'Vendor': 'vendor',
        'HP Cluster (SND Wajib Isi)': 'hp_cluster',
        'Status PO Cluster (SND Wajib Isi)': 'status_po'
    }

    df = df_raw.rename(columns=column_map)

    # Validasi kolom
    if not all(col in df.columns for col in column_map.values()):
        st.error("❌ Kolom yang dibutuhkan tidak ditemukan. Periksa file Excel Anda.")
        st.stop()

    df = df[list(column_map.values())].fillna("Unknown").astype(str)

    # Visualisasi distribusi fitur kategorikal
    st.subheader("📊 Distribusi Fitur")
    for col in ['topologi', 'vendor', 'hp_cluster']:
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f"Distribusi {col}")
        st.pyplot(fig)

    # Distribusi target
    st.subheader("🎯 Distribusi Target (status_po)")
    fig_target, ax_target = plt.subplots()
    df['status_po'].value_counts().plot(kind='bar', color='orange', ax=ax_target)
    st.pyplot(fig_target)

    # Label Encoding
    for col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Korelasi heatmap
    st.subheader("🔥 Korelasi Fitur (Heatmap)")
    fig_heat, ax_heat = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax_heat)
    st.pyplot(fig_heat)

    # Model & Evaluasi
    X = df.drop('status_po', axis=1)
    y = df['status_po']
    splits = [0.3, 0.2, 0.1]
    results = []

    for ratio in splits:
        split_label = f"{int((1 - ratio) * 100)}:{int(ratio * 100)}"
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
                "Split": split_label,
                "Akurasi": accuracy_score(y_test, y_pred),
                "Presisi": precision_score(y_test, y_pred, average='macro', zero_division=0),
                "Recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, average='macro', zero_division=0)
            })

    result_df = pd.DataFrame(results)
    st.subheader("📋 Tabel Evaluasi Model")
    st.dataframe(result_df)

    # Grafik perbandingan akurasi
    st.subheader("📈 Grafik Perbandingan Akurasi")
    fig_acc, ax_acc = plt.subplots()
    for model in result_df['Model'].unique():
        data = result_df[result_df['Model'] == model]
        ax_acc.plot(data['Split'], data['Akurasi'], marker='o', label=model)
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title("Akurasi Berdasarkan Rasio Split")
    ax_acc.set_ylabel("Akurasi")
    ax_acc.legend()
    st.pyplot(fig_acc)

    # ROC Curve (gunakan split 80:20 sebagai sampel)
    st.subheader("📉 ROC Curve (Split 80:20)")
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
    ax_roc.set_title("ROC Curve")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)
