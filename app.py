import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title="Prediksi PO", layout="wide")

st.title("📊 Prediksi Status PO")

# ================== Upload file ==================
uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"])
if uploaded_file is not None:
    # Baca data awal
    df_raw = pd.read_excel(uploaded_file)
    total_awal = len(df_raw)

    # Bersihkan kolom nama
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # Pastikan kolom target ada
    if "status_po" not in df.columns:
        st.error("Kolom 'status_po' tidak ditemukan di file yang diunggah.")
        st.stop()

    # Bersihkan kolom status_po
    df["status_po"] = df["status_po"].astype(str).str.strip().str.title()

    # Drop baris kosong
    df = df.dropna()

    total_setelah = len(df)

    st.markdown(f"**Jumlah Data Awal:** {total_awal} | **Setelah Preprocessing:** {total_setelah}")

    # One-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=False)

    # Cari nama kolom target yang mengandung "Tercapai"
    target_candidates = [col for col in df_encoded.columns if "Tercapai" in col]
    if not target_candidates:
        st.error("Tidak ditemukan kolom target yang mengandung kata 'Tercapai' setelah encoding.")
        st.write("Kolom yang ada:", list(df_encoded.columns))
        st.stop()

    target_col = target_candidates[0]  # ambil kolom pertama yang cocok

    # Pisahkan X dan y
    X = df_encoded.drop(target_col, axis=1, errors="ignore")
    y = df_encoded[target_col]

    # Split data
    split_ratio = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=split_ratio, random_state=42
    )

    # Info distribusi train/test
    train_tercapai = int((y_train == 1).sum())
    train_tidak = int((y_train == 0).sum())
    test_tercapai = int((y_test == 1).sum())
    test_tidak = int((y_test == 0).sum())

    st.markdown("### Distribusi Data Training & Testing")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Training Set:** {len(y_train)} data  \n- Tercapai: {train_tercapai}  \n- Tidak: {train_tidak}")
    with col2:
        st.markdown(f"**Testing Set:** {len(y_test)} data  \n- Tercapai: {test_tercapai}  \n- Tidak: {test_tidak}")

    # ================== Model Decision Tree ==================
    model_c45 = DecisionTreeClassifier(random_state=42)
    model_c45.fit(X_train, y_train)

    # Prediksi train & test
    y_pred_c45_train = model_c45.predict(X_train)
    y_pred_c45_test = model_c45.predict(X_test)

    # ================== Model Naive Bayes ==================
    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)

    y_pred_nb_train = model_nb.predict(X_train)
    y_pred_nb_test = model_nb.predict(X_test)

    # ================== Grafik ==================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Grafik train
    train_counts = pd.DataFrame({
        "Model": ["Decision Tree"]*2 + ["Naive Bayes"]*2,
        "Status": ["Tercapai", "Tidak"]*2,
        "Jumlah": [
            (y_pred_c45_train == 1).sum(),
            (y_pred_c45_train == 0).sum(),
            (y_pred_nb_train == 1).sum(),
            (y_pred_nb_train == 0).sum()
        ]
    })
    sns.barplot(data=train_counts, x="Model", y="Jumlah", hue="Status", ax=axes[0])
    axes[0].set_title("Prediksi Training Set")

    # Grafik test
    test_counts = pd.DataFrame({
        "Model": ["Decision Tree"]*2 + ["Naive Bayes"]*2,
        "Status": ["Tercapai", "Tidak"]*2,
        "Jumlah": [
            (y_pred_c45_test == 1).sum(),
            (y_pred_c45_test == 0).sum(),
            (y_pred_nb_test == 1).sum(),
            (y_pred_nb_test == 0).sum()
        ]
    })
    sns.barplot(data=test_counts, x="Model", y="Jumlah", hue="Status", ax=axes[1])
    axes[1].set_title("Prediksi Testing Set")

    st.pyplot(fig)

else:
    st.info("Silakan unggah file Excel terlebih dahulu.")
