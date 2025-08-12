import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import SMOTE but we'll handle runtime errors when using it
from imblearn.over_sampling import SMOTE
import io

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Dashboard Analisis C4.5 vs Naive Bayes", layout="wide")
sns.set_theme(style="whitegrid")

# DARK MODE CSS
dark_css = """
<style>
body { background-color: #121212; color: #E0E0E0; }
.sidebar .sidebar-content { background: #1E1E1E; }
.stButton > button { background-color: #4CAF50; color: white; border-radius: 8px; }
div[data-testid="stHorizontalBlock"] > div { border-radius: 12px; padding: 12px; }
h1, h2, h3, h4, h5 { color: #81C784; }
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# ----------------- TITLE -----------------
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>📊 Dashboard Analisis C4.5 vs Naive Bayes</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Prediksi Ketercapaian Target PO - MyRepublic</p>", unsafe_allow_html=True)
st.markdown("---")

# ----------------- SESSION STATE -----------------
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# ----------------- UPLOAD SECTION -----------------
if not st.session_state.file_uploaded:
    uploaded_file = st.file_uploader("🗂 Upload File Excel", type=["xlsx"])
    if uploaded_file is not None:
        st.session_state.file_uploaded = True
        st.session_state.uploaded_file = uploaded_file
        st.rerun()
else:
    uploaded_file = st.session_state.uploaded_file
    st.success(f"✅ File berhasil diunggah: {uploaded_file.name}")
    if st.button("🔄 Reset File"):
        st.session_state.file_uploaded = False
        st.rerun()

# ----------------- MAIN PROCESS -----------------
if st.session_state.file_uploaded:
    df_raw = pd.read_excel(uploaded_file)

    # INFO: jumlah data awal
    jumlah_awal = len(df_raw)

    # ----------------- SAFE PREPROCESSING (preserve fitur awal) -----------------
    # Rename sesuai mapping sebelumnya
    df = df_raw.rename(columns={
        'Topology': 'topologi',
        'Vendor': 'vendor',
        'HP Cluster\n(SND Wajib Isi)': 'hp_cluster',
        'Status PO Cluster (SND Wajib Isi)': 'status_po'
    })

    # Buang kolom yang kosong total (jika ada) untuk mencegah noise
    df = df.dropna(axis=1, how='all')

    # Hapus baris yang tidak mempunyai target (status_po) — kita butuh label untuk training
    if 'status_po' not in df.columns:
        st.error("Kolom 'Status PO Cluster (SND Wajib Isi)' tidak ditemukan setelah rename. Periksa header excel.")
        st.stop()
    df = df.dropna(subset=['status_po'])

    # Keep only the columns we need (keamanan: pastikan kolom ada)
    needed_cols = ['topologi', 'vendor', 'hp_cluster', 'status_po']
    available = [c for c in needed_cols if c in df.columns]
    if len(available) < 4:
        st.warning(f"Hanya kolom berikut yang tersedia setelah rename & drop: {available}. Pastikan Excel sesuai format.")
    # Select existing needed cols (earlier code required exactly these)
    df = df[[c for c in needed_cols if c in df.columns]]

    # INFO: jumlah setelah preprocess awal (sebelum filling)
    jumlah_setelah = len(df)
    st.info(f"📋 Jumlah data awal: **{jumlah_awal} baris**  \n🧹 Setelah preprocessing awal: **{jumlah_setelah} baris**")

    # Normalisasi & label
    df['status_po'] = df['status_po'].astype(str).str.lower().str.strip()
    # label: 1 jika 'done' (case-insensitive), else 0
    df['label'] = df['status_po'].apply(lambda x: 1 if x == 'done' else 0)

    # Isi NaN pada fitur numerik hp_cluster dengan median (agar SMOTE & scaler aman)
    if 'hp_cluster' in df.columns:
        median_hp = df['hp_cluster'].median()
        df['hp_cluster'] = df['hp_cluster'].fillna(median_hp)
    else:
        # jika kolom hp_cluster tidak ada, buat default 0 (tapi beri peringatan)
        st.warning("Kolom 'HP Cluster' tidak ditemukan, membuat kolom 'hp_cluster' default 0.")
        df['hp_cluster'] = 0.0

    # Encode kategorikal (safe: convert to str before label encoding)
    df['topologi_enc'] = LabelEncoder().fit_transform(df['topologi'].astype(str))
    df['vendor_enc'] = LabelEncoder().fit_transform(df['vendor'].astype(str))

    # Normalisasi hp_cluster ke hp_cluster_norm
    df['hp_cluster_norm'] = MinMaxScaler().fit_transform(df[['hp_cluster']].astype(float))

    # Sidebar controls (preserve original)
    st.sidebar.header("⚙️ Pengaturan Analisis")
    split_option = st.sidebar.radio("Pilih Rasio Split Data", ["80:20", "70:30", "90:10"])
    metric_option = st.sidebar.radio("Pilih Metrik Evaluasi", ["Accuracy", "Precision", "Recall", "F1-score"])
    use_smote = st.sidebar.checkbox("", value=True)

    split_map = {"80:20": 0.2, "70:30": 0.3, "90:10": 0.1}
    split_ratio = split_map[split_option]

    # Siapkan X & y (pastikan numeric)
    X = df[['topologi_enc', 'vendor_enc', 'hp_cluster_norm']].astype(float)
    y = df['label']

    # SPLIT dulu (dengan stratify) sehingga kita bisa tampilkan distribusi train/test
    # jika stratify gagal (misal semua label sama), fallback tanpa stratify
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=split_ratio, random_state=42)
    except ValueError as e:
        st.warning(f"Stratify gagal karena distribusi label: {e}. Melakukan split tanpa stratify.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

    # Tampilkan distribusi data train & test
    st.markdown("### 📌 Distribusi Data")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"**Training Set:** {len(y_train)} data")
        st.markdown(f"- Tercapai (label=1): **{int((y_train==1).sum())}**")
        st.markdown(f"- Tidak (label=0): **{int((y_train==0).sum())}**")
    with col_info2:
        st.markdown(f"**Testing Set:** {len(y_test)} data")
        st.markdown(f"- Tercapai (label=1): **{int((y_test==1).sum())}**")
        st.markdown(f"- Tidak (label=0): **{int((y_test==0).sum())}**")

    # ----------------- SMOTE SAFETY & TRAINING -----------------
    # Prepare training sets for C4.5 (use original X_train) and NB (optionally resampled)
    X_train_for_c45 = X_train.copy()
    y_train_for_c45 = y_train.copy()

    X_train_for_nb = X_train.copy()
    y_train_for_nb = y_train.copy()
    smote_used = False
    smote_error_msg = None

    if use_smote:
        # Only attempt SMOTE when there are at least 2 classes in y_train
        vc = y_train.value_counts()
        if vc.size > 1 and vc.min() > 0:
            # SMOTE requires that minority class has at least k_neighbors+1 samples.
            # Default k_neighbors=5, adjust if minority too small.
            min_count = int(vc.min())
            if min_count < 2:
                # too few samples to resample
                st.warning("Kelas minoritas terlalu kecil untuk SMOTE (min count < 2). SMOTE dilewati untuk Naive Bayes.")
            else:
                k_neighbors = 5
                if min_count - 1 < k_neighbors:
                    k_neighbors = max(1, min_count - 1)
                try:
                    # Try to run SMOTE; catch any runtime/version error and fallback
                    sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
                    X_train_res, y_train_res = sm.fit_resample(X_train_for_nb, y_train_for_nb)
                    X_train_for_nb = pd.DataFrame(X_train_res, columns=X_train.columns)
                    y_train_for_nb = pd.Series(y_train_res)
                    smote_used = True
                except Exception as ex:
                    smote_error_msg = str(ex)
                    
            st.warning("")
    else:
        st.info("")

    # Convert training sets to arrays (scaler will be applied to training for NB)
    scaler_nb = MinMaxScaler()
    try:
        X_train_nb_scaled = scaler_nb.fit_transform(X_train_for_nb)
        X_test_scaled = scaler_nb.transform(X_test)
    except Exception:
        # fallback if something unexpected: cast explicitly then scale
        X_train_nb_scaled = scaler_nb.fit_transform(X_train_for_nb.astype(float))
        X_test_scaled = scaler_nb.transform(X_test.astype(float))

    # Training model (letakkan di spinner)
    with st.spinner("🔄 Training model... Mohon tunggu"):
        # C4.5 (Decision Tree) - using original X_train (no SMOTE), consistent with original behavior
        model_c45 = DecisionTreeClassifier(criterion='entropy', random_state=42)
        model_c45.fit(X_train_for_c45, y_train_for_c45)
        # prediksi test & train
        y_pred_c45 = model_c45.predict(X_test)
        y_pred_c45_train = model_c45.predict(X_train_for_c45)

        # Naive Bayes - trained on (possibly resampled) X_train_for_nb (scaled)
        model_nb = GaussianNB()
        model_nb.fit(X_train_nb_scaled, y_train_for_nb)
        # prediksi test & train
        y_pred_nb = model_nb.predict(X_test_scaled)
        y_pred_nb_train = model_nb.predict(X_train_nb_scaled)

    # Evaluasi metrik (gunakan zero_division=0 untuk menghindari error)
    def evaluate(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-score": f1_score(y_true, y_pred, zero_division=0)
        }

    c45_result = evaluate(y_test, y_pred_c45)
    nb_result = evaluate(y_test, y_pred_nb)

    df_eval = pd.DataFrame([
        {"Model": "C4.5", **c45_result},
        {"Model": "Naive Bayes", **nb_result}
    ])

    best = df_eval.sort_values(by=metric_option, ascending=False).iloc[0]

    # Confusion Matrix (testing)
    cm_c45 = confusion_matrix(y_test, y_pred_c45)
    cm_nb = confusion_matrix(y_test, y_pred_nb)

    # ----------------- HASIL PREDIKSI PO -----------------
    st.markdown("### 🎯 Hasil Prediksi PO Tercapai & Tidak Tercapai (Training vs Testing)")
    colA, colB = st.columns(2)

    # Hitung jumlah prediksi (testing)
    c45_tercapai_test = int((y_pred_c45 == 1).sum())
    c45_tidak_test = int((y_pred_c45 == 0).sum())
    nb_tercapai_test = int((y_pred_nb == 1).sum())
    nb_tidak_test = int((y_pred_nb == 0).sum())

    # Hitung jumlah prediksi (training)
    c45_tercapai_train = int((y_pred_c45_train == 1).sum())
    c45_tidak_train = int((y_pred_c45_train == 0).sum())
    nb_tercapai_train = int((y_pred_nb_train == 1).sum())
    nb_tidak_train = int((y_pred_nb_train == 0).sum())

    # ---- C4.5 ----
    with colA:
        st.markdown("#### 🔴 C4.5")
        sub1, sub2 = st.columns([1, 1])
        with sub1:
            st.markdown("**Training**")
            st.markdown(f"- **Tercapai:** {c45_tercapai_train}  \n- **Tidak:** {c45_tidak_train}")
            fig_c45_train, ax_c45_train = plt.subplots(figsize=(2.6, 2.2))
            sns.barplot(x=['Tercapai', 'Tidak'], y=[c45_tercapai_train, c45_tidak_train],
                        palette=['#4CAF50', '#E53935'], ax=ax_c45_train)
            for i, v in enumerate([c45_tercapai_train, c45_tidak_train]):
                ax_c45_train.text(i, v + max(1, v * 0.01), str(v), ha='center', fontsize=8)
            ax_c45_train.set_ylabel("")
            ax_c45_train.tick_params(axis='both', labelsize=8)
            st.pyplot(fig_c45_train)
        with sub2:
            st.markdown("**Testing**")
            st.markdown(f"- **Tercapai:** {c45_tercapai_test}  \n- **Tidak:** {c45_tidak_test}")
            fig_c45_test, ax_c45_test = plt.subplots(figsize=(2.6, 2.2))
            sns.barplot(x=['Tercapai', 'Tidak'], y=[c45_tercapai_test, c45_tidak_test],
                        palette=['#4CAF50', '#E53935'], ax=ax_c45_test)
            for i, v in enumerate([c45_tercapai_test, c45_tidak_test]):
                ax_c45_test.text(i, v + max(1, v * 0.01), str(v), ha='center', fontsize=8)
            ax_c45_test.set_ylabel("")
            ax_c45_test.tick_params(axis='both', labelsize=8)
            st.pyplot(fig_c45_test)

    # ---- Naive Bayes ----
    with colB:
        st.markdown("#### 🔵 Naive Bayes")
        sub3, sub4 = st.columns([1, 1])
        with sub3:
            st.markdown("**Training**")
            # show whether training used SMOTE (informational)
            if smote_used:
                st.markdown(f"- **(SMOTE digunakan)**")
           
            with st.container():
                st.markdown(f"- **Tercapai:** {nb_tercapai_train}  \n- **Tidak:** {nb_tidak_train}")
                fig_nb_train, ax_nb_train = plt.subplots(figsize=(2.6, 2.2))
                sns.barplot(x=['Tercapai', 'Tidak'], y=[nb_tercapai_train, nb_tidak_train],
                            palette=['#4CAF50', '#E53935'], ax=ax_nb_train)
                for i, v in enumerate([nb_tercapai_train, nb_tidak_train]):
                    ax_nb_train.text(i, v + max(1, v * 0.01), str(v), ha='center', fontsize=8)
                ax_nb_train.set_ylabel("")
                ax_nb_train.tick_params(axis='both', labelsize=8)
                st.pyplot(fig_nb_train)
        with sub4:
            st.markdown("**Testing**")
            st.markdown(f"- **Tercapai:** {nb_tercapai_test}  \n- **Tidak:** {nb_tidak_test}")
            fig_nb_test, ax_nb_test = plt.subplots(figsize=(2.6, 2.2))
            sns.barplot(x=['Tercapai', 'Tidak'], y=[nb_tercapai_test, nb_tidak_test],
                        palette=['#4CAF50', '#E53935'], ax=ax_nb_test)
            for i, v in enumerate([nb_tercapai_test, nb_tidak_test]):
                ax_nb_test.text(i, v + max(1, v * 0.01), str(v), ha='center', fontsize=8)
            ax_nb_test.set_ylabel("")
            ax_nb_test.tick_params(axis='both', labelsize=8)
            st.pyplot(fig_nb_test)

    # ----------------- TABEL HASIL UNTUK SEMUA SPLIT -----------------
    st.markdown("### 📑 Hasil Prediksi PO untuk Semua Split Data")
    split_ratios = {"70:30": 0.3, "80:20": 0.2, "90:10": 0.1}
    prediksi_list = []

    for name, ratio in split_ratios.items():
        # split (try stratify first)
        try:
            X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y, stratify=y, test_size=ratio, random_state=42)
        except ValueError:
            X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y, test_size=ratio, random_state=42)

        # Apply same SMOTE-safety logic per split for NB & C45 comparison
        X_train_nb_s = X_train_s.copy()
        y_train_nb_s = y_train_s.copy()
        smote_applied_s = False
        vc_s = y_train_s.value_counts()
        if use_smote and vc_s.size > 1 and vc_s.min() > 1:
            k_neigh = 5 if vc_s.min() - 1 >= 5 else max(1, vc_s.min() - 1)
            try:
                Xr, yr = SMOTE(random_state=42, k_neighbors=k_neigh).fit_resample(X_train_nb_s, y_train_nb_s)
                X_train_nb_s = pd.DataFrame(Xr, columns=X_train_s.columns)
                y_train_nb_s = pd.Series(yr)
                smote_applied_s = True
            except Exception:
                smote_applied_s = False

        # Train C4.5 on training split (without SMOTE resample to match original)
        model_c45_s = DecisionTreeClassifier(criterion='entropy', random_state=42)
        model_c45_s.fit(X_train_s, y_train_s)
        y_pred_c45_s = model_c45_s.predict(X_test_s)

        # Train NB on training split (resampled if applied)
        scaler_tmp = MinMaxScaler()
        X_train_nb_s_scaled = scaler_tmp.fit_transform(X_train_nb_s.astype(float))
        X_test_s_scaled = scaler_tmp.transform(X_test_s.astype(float))
        model_nb_s = GaussianNB()
        model_nb_s.fit(X_train_nb_s_scaled, y_train_nb_s)
        y_pred_nb_s = model_nb_s.predict(X_test_s_scaled)

        prediksi_list.append({"Split": name, "Model": "C4.5", "Tercapai": int((y_pred_c45_s == 1).sum()), "Tidak Tercapai": int((y_pred_c45_s == 0).sum())})
        prediksi_list.append({"Split": name, "Model": "Naive Bayes", "Tercapai": int((y_pred_nb_s == 1).sum()), "Tidak Tercapai": int((y_pred_nb_s == 0).sum())})

    df_split_prediksi = pd.DataFrame(prediksi_list)
    st.dataframe(df_split_prediksi.style.set_properties(**{'text-align': 'center'}).highlight_max(subset=["Tercapai"], color="lightgreen"))

    # ----------------- LAYOUT ANALISIS & GRAFIK -----------------
    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div style="background:#263238; padding:20px; border-radius:12px; color:white; box-shadow:0 4px 8px rgba(0,0,0,0.3);">
        <h3 style='color:#4CAF50;'>📌 Ringkasan Analisis</h3>
        <p><b>Metrik:</b> {metric_option}</p>
        <p><b>Model Terbaik:</b> <span style='color:#81C784;'>{best['Model']}</span></p>
        <p><b>Skor:</b> {best[metric_option]:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

        csv = df_eval.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Hasil (CSV)", data=csv, file_name="hasil_evaluasi.csv", mime="text/csv")

    with col2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        sns.barplot(data=df_eval, x='Model', y=metric_option, palette="viridis", ax=axes[0])
        axes[0].set_ylim(0, 1)
        axes[0].set_title(f"Perbandingan {metric_option}")
        for i, val in enumerate(df_eval[metric_option]):
            axes[0].text(i, val + 0.02, f"{val:.2f}", ha='center', fontsize=9)

        sns.heatmap(cm_c45, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_title("C4.5")

        sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', ax=axes[2])
        axes[2].set_title("Naive Bayes")

        plt.tight_layout()
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.download_button("⬇️ Download Grafik (PNG)", data=buf, file_name="grafik_dashboard.png", mime="image/png")

    # ----------------- TABEL -----------------
    st.markdown("<h3 style='color:#81C784;'>📄 Tabel Evaluasi Lengkap</h3>", unsafe_allow_html=True)
    st.dataframe(df_eval.style.highlight_max(axis=0, color='lightgreen'))



