import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Tema Seaborn
sns.set_theme(style="whitegrid")

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Algoritma C4.5 vs Naive Bayes", layout="wide")

# Judul Utama
st.title("📊 Analisis Perbandingan Algoritma C4.5 dan Naive Bayes")
st.markdown("### Prediksi Ketercapaian Target PO - MyRepublic")
st.markdown("---")

# Upload File
uploaded_file = st.file_uploader("🗂 Upload File Excel", type=["xlsx"])

if uploaded_file is not None:
    df_raw = pd.read_excel(uploaded_file)

    # Preprocessing
    df = df_raw.rename(columns={
        'Topology': 'topologi',
        'Vendor': 'vendor',
        'HP Cluster\n(SND Wajib Isi)': 'hp_cluster',
        'Status PO Cluster (SND Wajib Isi)': 'status_po'
    })[['topologi', 'vendor', 'hp_cluster', 'status_po']].dropna()

    df['status_po'] = df['status_po'].str.lower().str.strip()
    df['label'] = df['status_po'].apply(lambda x: 1 if x == 'done' else 0)
    df['topologi_enc'] = LabelEncoder().fit_transform(df['topologi'].astype(str))
    df['vendor_enc'] = LabelEncoder().fit_transform(df['vendor'].astype(str))
    df['hp_cluster_norm'] = MinMaxScaler().fit_transform(df[['hp_cluster']])

    # Sidebar Controls
    st.sidebar.header("⚙️ Pengaturan Analisis")
    split_option = st.sidebar.selectbox("Pilih Rasio Split Data", ["80:20", "70:30", "90:10"])
    metric_option = st.sidebar.selectbox("Pilih Metrik Evaluasi", ["Accuracy", "Precision", "Recall", "F1-score"])

    split_map = {"80:20": 0.2, "70:30": 0.3, "90:10": 0.1}
    split_ratio = split_map[split_option]

    X = df[['topologi_enc', 'vendor_enc', 'hp_cluster_norm']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=split_ratio, random_state=42)

    # Training Model
    model_c45 = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model_c45.fit(X_train, y_train)
    y_pred_c45 = model_c45.predict(X_test)

    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    y_pred_nb = model_nb.predict(X_test)

    # Evaluasi
    def evaluate(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-score": f1_score(y_true, y_pred)
        }

    c45_result = evaluate(y_test, y_pred_c45)
    nb_result = evaluate(y_test, y_pred_nb)

    df_eval = pd.DataFrame([
        {"Model": "C4.5", **c45_result},
        {"Model": "Naive Bayes", **nb_result}
    ])

    # Tampilan Analisis
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📈 Grafik Performa Model")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        sns.barplot(data=df_eval, x='Model', y=metric_option, palette="viridis", ax=ax1)
        ax1.set_ylim(0, 1)
        ax1.set_title(f"Perbandingan {metric_option}", fontsize=14, weight='bold')
        for i, val in enumerate(df_eval[metric_option]):
            ax1.text(i, val + 0.02, f"{val:.2f}", ha='center', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        st.markdown("### 📊 Confusion Matrix")
        fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred_c45), annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title("C4.5", fontsize=12)
        sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title("Naive Bayes", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig2)

    # Tabel Evaluasi
    st.markdown("### 📄 Tabel Evaluasi Lengkap")
    st.dataframe(df_eval.style.highlight_max(axis=0, color='lightgreen'))

    # Kesimpulan Otomatis
    best = df_eval.sort_values(by=metric_option, ascending=False).iloc[0]
    st.success(f"📌 Berdasarkan metrik **{metric_option}**, model terbaik adalah **{best['Model']}** dengan skor **{best[metric_option]:.4f}**.")
