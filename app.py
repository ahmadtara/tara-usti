import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import io
import base64

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Analisis C4.5 vs Naive Bayes", layout="wide")

# --- CSS DARK MODE ---
dark_css = """
<style>
body {
    background-color: #121212;
    color: #E0E0E0;
}
.sidebar .sidebar-content {
    background: #1E1E1E;
}
.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
}
div[data-testid="stHorizontalBlock"] > div {
    border-radius: 12px;
    padding: 12px;
}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# Judul Utama
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>📊 Dashboard Analisis C4.5 vs Naive Bayes</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Prediksi Ketercapaian Target PO - MyRepublic</p>", unsafe_allow_html=True)
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
    split_option = st.sidebar.radio("Pilih Rasio Split Data", ["80:20", "70:30", "90:10"])
    metric_option = st.sidebar.radio("Pilih Metrik Evaluasi", ["Accuracy", "Precision", "Recall", "F1-score"])

    split_map = {"80:20": 0.2, "70:30": 0.3, "90:10": 0.1}
    split_ratio = split_map[split_option]

    X = df[['topologi_enc', 'vendor_enc', 'hp_cluster_norm']]
    y = df['label']

    # --- TRAINING DENGAN LOADING ANIMASI ---
    with st.spinner("🔄 Training model... Mohon tunggu"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=split_ratio, random_state=42)

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

    # Hasil terbaik
    best = df_eval.sort_values(by=metric_option, ascending=False).iloc[0]

    # --- DASHBOARD ---
    col1, col2 = st.columns([1, 2])

    # Card Ringkasan
    with col1:
        st.markdown("""
        <div style="background:#263238; padding:20px; border-radius:12px; color:white; box-shadow:0 4px 8px rgba(0,0,0,0.3);">
        <h3 style='color:#4CAF50;'>📌 Ringkasan Analisis</h3>
        <p><b>Metrik:</b> {}</p>
        <p><b>Model Terbaik:</b> <span style='color:#81C784;'>{}</span></p>
        <p><b>Skor:</b> {:.4f}</p>
        </div>
        """.format(metric_option, best['Model'], best[metric_option]), unsafe_allow_html=True)

        # Download CSV
        csv = df_eval.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Hasil (CSV)", data=csv, file_name="hasil_evaluasi.csv", mime="text/csv")

    # Grafik + Confusion Matrix
    with col2:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.barplot(data=df_eval, x='Model', y=metric_option, palette="viridis", ax=axes[0])
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Perbandingan {}".format(metric_option))
        for i, val in enumerate(df_eval[metric_option]):
            axes[0].text(i, val + 0.02, f"{val:.2f}", ha='center')

        sns.heatmap(confusion_matrix(y_test, y_pred_c45), annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_title("C4.5")

        sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Greens', ax=axes[2])
        axes[2].set_title("Naive Bayes")

        plt.tight_layout()
        st.pyplot(fig)

        # Download Grafik
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.download_button("⬇️ Download Grafik (PNG)", data=buf, file_name="grafik_dashboard.png", mime="image/png")

    # Tabel Evaluasi
    st.markdown("<h3 style='color:#81C784;'>📄 Tabel Evaluasi Lengkap</h3>", unsafe_allow_html=True)
    st.dataframe(df_eval.style.highlight_max(axis=0, color='lightgreen'))
